import argparse
from gpu_helpers import print_details
from static_globals import *
from datasets import load_dataset
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import random

parser=argparse.ArgumentParser()

parser.add_argument("--method_name",type=str,default=PROGRESSIVE)
parser.add_argument("--dataset",type=str,default="jlbaker361/league-hard-prompt")
parser.add_argument("--image_field_name",type=str,default="tile")
parser.add_argument("--text_field_name",type=str,default="subject")
parser.add_argument("--limit",type=int,default=10)
parser.add_argument("--gradient_accumulation_steps",default=8,type=int)
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="distillation")
parser.add_argument("--initial_num_inference_steps",type=int,default=32)
parser.add_argument("--final_num_inference_steps",type=int,default=1)
parser.add_argument("--pretrained_path",type=str,default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--ip_weight_name",type=str,default="ip-adapter_sd15.bin")
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--use_lora",action="store_true")
parser.add_argument("--lr",type=float,default=0.01)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--convergence_threshold",type=float,default=0.001,help="stop training once student and teacher are this close")
parser.add_argument("--do_classifier_free_guidance",action="store_true")
parser.add_argument("--seed",type=int,default=123)
parser.add_argument("--prediction_method",type=str,default=REVERSE)
#TODO set sampler as arg
#TODO noise prediction vs x prediction
'''
Predicting x directly.

Predicting both x and epsilon, via separate output channels {x˜θ(zt), epsilon ˜θ(zt)} of the neural network, and then merging the predictions via 
xˆ = σ2t x˜θ(zt) + αt(zt − σt epsilon˜θ(zt)), thus
smoothly interpolating between predicting x directly and predicting via epsilon.


'''

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data=load_dataset(args.dataset,split="train")
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    decriptor_list=[" picture of {}"," image of {}"," depiction of {}", " photo of {}", "{}"]
    location_list=[" at the beach", " in the jungle", " in the city", " at school", " in an office", " "]
    activity_list=[" skydiving "," eating ", " writing ", " standing ", " making pottery ", " talking ", " "]
    for i, row in enumerate(data):
        if i>args.limit:
            break
        image=row[args.image_field_name]
        subject=row[args.text_field_name]
        training_prompt_list=[]
        for descriptor in decriptor_list:
            for location in location_list:
                for activity in activity_list:
                    training_prompt_list.append(f"{descriptor} {activity} {location}".format(subject))
        print(training_prompt_list)
        if args.method_name==PROGRESSIVE:
            print("effective batch size = ",args.batch_size* args.gradient_accumulation_steps)
            teacher_pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_path)
            student_pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_path)
            for pipeline in [teacher_pipeline,student_pipeline]:
                pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name)
                pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
                pipeline.scheduler.set_timesteps(args.initial_num_inference_steps)
            
            
            #TODO: IP adapter preparation
            ip_adapter_image_embeds = student_pipeline.prepare_ip_adapter_image_embeds(
                image,
                None,
                accelerator.device,
                1,
                args.do_classifier_free_guidance,
            )
            added_cond_kwargs ={"image_embeds":ip_adapter_image_embeds}

            i=0 #prompt stuff preparation
            while len(training_prompt_list)%args.batch_size!=0:
                training_prompt_list.append(training_prompt_list[i])
                i+=1
            positive_prompt_list=[]
            negative_prompt_list=[]
            for positive,negative in [teacher_pipeline.encode_prompt(prompt=prompt,negative_prompt=NEGATIVE,do_classifier_free_guidance=args.do_classifier_free_guidance) for prompt in  training_prompt_list]:
                positive_prompt_list.append(positive)
                negative_prompt_list.append(negative)
            negative_prompt_list_batched=[torch.cat(negative_prompt_list[i:i+args.batch_size]) for i in range(0,len(negative_prompt_list),args.batch_size)]
            positive_prompt_list_batched=[torch.cat(positive_prompt_list[i:i+args.batch_size]) for i in range(0,len(positive_prompt_list), args.batch_size)]

            student_steps=args.initial_num_inference_steps//2
            num_channels_latents = teacher_pipeline.unet.config.in_channels
            while student_steps>=args.final_num_inference_steps:
                teacher_pipeline=student_pipeline
                teacher_pipeline.unet.requires_grad_(False)
                student_pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_path)
                student_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name)
                student_pipeline.unet.load_state_dict(teacher_pipeline.unet.state_dict())
                student_pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
                student_pipeline.scheduler.set_timesteps(student_steps)
                student_pipeline.unet.requires_grad_(True)
                student_pipeline.unet.to(accelerator.device)
                teacher_pipeline.unet.to(accelerator.device)

                trainable_parameters=filter(lambda p: p.requires_grad, student_pipeline.unet.parameters())
                #print(trainable_parameters)
                optimizer = torch.optim.AdamW(
                    trainable_parameters,
                    lr=args.lr,
                    betas=(0.9, 0.999),
                    weight_decay=0.01,
                    eps=0.00000001)
                total_steps=0
                for e in range(args.epochs):
                    avg_loss=0.0
                    if args.prediction_method==REVERSE:
                        for positive,negative in zip(positive_prompt_list_batched, negative_prompt_list_batched):
                            #TODO prepare and clone latents
                            positive=positive.to(accelerator.device)
                            negative=negative.to(accelerator.device)
                            with accelerator.accumulate(student_pipeline.unet):
                                pass
                                
                                #iterate through timesteps
                                #STUDENT:
                                # expand the latents if we are doing classifier free guidance
                                # predict the noise residual
                                # compute the previous noisy sample x_t -> x_t-1
                                
                                #TEACHER:
                                # expand the latents if we are doing classifier free guidance
                                # predict the noise residual
                                # compute the previous noisy sample x_t -> x_t-1
                                #then REPEAT
                                #compute loss
                                
                                #logging
                                #optimization step
                        #check if avg loss<convergence
                #metrics


if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!!!")