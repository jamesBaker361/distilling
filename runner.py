import argparse
from gpu_helpers import print_details
from static_globals import *
from datasets import load_dataset
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import random
from distillation_helpers import reverse_step, clone_pipeline
import torch.nn.functional as F
import time
from adapter_helpers import better_load_ip_adapter
from datetime import datetime

# getting the current date and time
current_datetime = datetime.now()

torch.autograd.set_detect_anomaly(True)

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
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--use_negative_prompt",action="store_true")
parser.add_argument("--guidance_scale",type=float,default=7.5)
parser.add_argument("--shuffle",action="store_true")
parser.add_argument("--pretrain_noise_pipeline",action="store_true")
parser.add_argument("--use_ip_adapter",action="store_true")
#TODO set sampler as arg
#TODO noise prediction vs x prediction
#TODO SNR coefficien
'''
Predicting x directly.

Predicting both x and epsilon, via separate output channels {x˜θ(zt), epsilon ˜θ(zt)} of the neural network, and then merging the predictions via 
xˆ = σ2t x˜θ(zt) + αt(zt − σt epsilon˜θ(zt)), thus
smoothly interpolating between predicting x directly and predicting via epsilon.


'''

def main(args):
    current_date_time = current_datetime.strftime("%m/%d/%Y, %H:%M:%S")
    print("current date and time = ",current_date_time)
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
                    training_prompt_list.append(f"{descriptor} {activity} {location}".format(subject).replace("  "," ").replace("  "," "))
        #print(training_prompt_list)
        generator=torch.Generator(accelerator.device)
        generator.manual_seed(args.seed)

        effective_batch_size=args.batch_size* args.gradient_accumulation_steps
        print("effective batch size = ",effective_batch_size)

        teacher_pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_path)
        if args.use_ip_adapter:
            teacher_pipeline=better_load_ip_adapter(
                teacher_pipeline,"h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name
            )
            #teacher_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name)
            teacher_pipeline("do this to help instantiate proerties",num_inference_steps=1,ip_adapter_image=image)
        else:
            teacher_pipeline("do this to help instantiate proerties",num_inference_steps=1)
        teacher_pipeline.scheduler=DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
        teacher_pipeline.scheduler.set_timesteps(args.initial_num_inference_steps)

        i=0 #prompt stuff preparation
        while len(training_prompt_list)%args.batch_size!=0:
            training_prompt_list.append(training_prompt_list[i])
            i+=1
        positive_prompt_list=[]
        negative_prompt_list=[]
        negative_prompt=" "
        if args.use_negative_prompt:
            negative_prompt=NEGATIVE
        for positive,negative in [teacher_pipeline.encode_prompt(prompt=prompt,negative_prompt=negative_prompt,do_classifier_free_guidance=args.do_classifier_free_guidance,device="cpu",num_images_per_prompt=1) for prompt in  training_prompt_list]:
            #print(type(positive),type(negative))
            positive_prompt_list.append(positive)
            negative_prompt_list.append(negative)
        print(len(positive_prompt_list), len(negative_prompt_list))
        if args.do_classifier_free_guidance:
            negative_prompt_list_batched=[torch.cat(negative_prompt_list[i:i+args.batch_size]) for i in range(0,len(negative_prompt_list),args.batch_size)]
        else:
            negative_prompt_list_batched=[negative_prompt_list[i:i+args.batch_size] for i in range(0,len(negative_prompt_list),args.batch_size)]
        positive_prompt_list_batched=[torch.cat(positive_prompt_list[i:i+args.batch_size]) for i in range(0,len(positive_prompt_list), args.batch_size)]
        if args.use_ip_adapter:
            ip_adapter_image_embeds = teacher_pipeline.prepare_ip_adapter_image_embeds(
                    image,
                    None,
                    "cpu",
                    1,
                    args.do_classifier_free_guidance,
                )[0]
            added_cond_kwargs ={"image_embeds":[ip_adapter_image_embeds.to(accelerator.device)]}
        else:
            added_cond_kwargs ={}
        print("len prompt list",len(positive_prompt_list))
        print("len batched ",len(positive_prompt_list_batched))
        if args.method_name==PROGRESSIVE:
            student_pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_path)
            if args.use_ip_adapter:
                student_pipeline=better_load_ip_adapter(
                    student_pipeline,"h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name
                )
                #student_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name)
                student_pipeline("do this to help instantiate proerties",num_inference_steps=1,ip_adapter_image=image)
            else:
                student_pipeline("do this to help instantiate proerties",num_inference_steps=1)
            student_pipeline.scheduler=DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
            student_pipeline.scheduler.set_timesteps(args.initial_num_inference_steps)
            student_pipeline.unet=student_pipeline.unet.to(accelerator.device)
            
            student_steps=args.initial_num_inference_steps//2
            num_channels_latents = teacher_pipeline.unet.config.in_channels
            while student_steps>=args.final_num_inference_steps:
                accelerator.gradient_accumulation_steps=min(accelerator.gradient_accumulation_steps,student_steps )
                print("effective batch size ",accelerator.gradient_accumulation_steps * args.batch_size)
                teacher_pipeline=student_pipeline
                teacher_pipeline.unet.requires_grad_(False)
                student_pipeline=StableDiffusionPipeline.from_pretrained(args.pretrained_path)
                if args.use_ip_adapter:
                    student_pipeline=better_load_ip_adapter(
                        student_pipeline,"h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name
                    )
                    #student_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name)
                    student_pipeline("do this to help instantiate proerties",num_inference_steps=1,ip_adapter_image=image)
                else:
                    student_pipeline("do this to help instantiate proerties",num_inference_steps=1)
                student_pipeline.unet.load_state_dict(teacher_pipeline.unet.state_dict())
                student_pipeline.scheduler=DDIMScheduler.from_config(teacher_pipeline.scheduler.config)
                student_pipeline.scheduler.set_timesteps(student_steps)
                student_pipeline.unet.requires_grad_(True)
                student_pipeline.unet=student_pipeline.unet.to(accelerator.device)

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
                    start=time.time()
                    epoch_loss=0.0
                    if args.prediction_method==REVERSE:
                        for positive,negative in zip(positive_prompt_list_batched, negative_prompt_list_batched):
                            #with accelerator.accumulate(student_pipeline.unet):
                            avg_loss=0.0
                            #TODO prepare and clone latents
                            student_latents = student_pipeline.vae.config.scaling_factor * student_pipeline.prepare_latents(
                                args.batch_size,
                                num_channels_latents,
                                args.size,
                                args.size,
                                positive.dtype,
                                accelerator.device,
                                generator)
                            teacher_latents=student_latents.clone()
                            teacher_latents_plus=student_latents.clone()
                            positive=positive.to(accelerator.device)
                            #print("latennts size",student_latents.size())
                            
                            if args.do_classifier_free_guidance:
                                negative=negative.to(accelerator.device)
                                prompt_embeds = torch.cat([negative, positive])
                            else:
                                prompt_embeds=positive
                            for student_i in range(len(student_pipeline.scheduler.timesteps)):
                                with accelerator.accumulate(student_pipeline.unet):
                                    #print("prompt embeds size",prompt_embeds.size())

                                    start_latents=teacher_latents_plus.clone()

                                    student_t=student_pipeline.scheduler.timesteps[student_i]
                                    teacher_i=(2*student_i)-1
                                    teacher_t=teacher_pipeline.scheduler.timesteps[teacher_i]
            
                                    student_latents=reverse_step(args,student_t,student_pipeline,start_latents,prompt_embeds, added_cond_kwargs)
                                    
                                    teacher_latents=reverse_step(args,teacher_t, teacher_pipeline, start_latents, prompt_embeds, added_cond_kwargs)
                                    teacher_t_plus=teacher_pipeline.scheduler.timesteps[teacher_i+1]
                                    teacher_latents_plus=reverse_step(args,teacher_t_plus, teacher_pipeline, teacher_latents, prompt_embeds, added_cond_kwargs)
                                    print(student_t, teacher_t, teacher_t_plus)
                                    #compute loss
                                    
                                    loss=F.mse_loss(teacher_latents_plus,student_latents,reduction="mean")
                                    avg_loss+=loss.detach().cpu().numpy()/effective_batch_size
                                    print(avg_loss)
                                    accelerator.backward(loss,retain_graph=True)
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    #avg_loss+=loss.detach().cpu().numpy()/effective_batch_size
                            accelerator.log({
                                "avg_loss_per_step_per_sample":avg_loss
                            })
                            epoch_loss+=avg_loss
                    accelerator.log({
                        "avg_loss_per_step_per_epoch": epoch_loss/(e+1)
                    })
                    end=time.time()
                    print(f"epochs {e} ended after {end-start} seconds = {(end-start)/3600} hours")
                    #check if epoch loss<convergence
                    if epoch_loss/(e+1)<args.convergence_threshold:
                        break
                student_steps=student_steps//2
                accelerator.free_memory()
                torch.cuda.empty_cache()
                #metrics
        elif args.method_name==CYCLE_GAN:
            noise_list=[]
            image_list=[]

            for prompt in training_prompt_list:
                noise_latents=teacher_pipeline.prepare_latents(
                                    1,
                                    num_channels_latents,
                                    args.size,
                                    args.size,
                                    positive.dtype,
                                    "cpu",
                                    generator)
                noise_list.append(noise_latents)
                image_latents=teacher_pipeline(prompt,latents=noise_latents,
                                               num_inference_steps=args.initial_num_inference_steps,
                                               negative_prompt=negative_prompt,ip_adapter_image=image,output_type="latent")
                image_list.append(image_latents)
            if args.shuffle:
                random.shuffle(image_list)
            noise_list_batched=[noise[i:i+args.batch_size] for noise in noise_list]
            image_list_batched=[image[i:i+args.batch_size] for image in image_list]

            image_pipeline=teacher_pipeline
            noise_pipeline=clone_pipeline(image_pipeline)
        elif args.method_name==TRACT:
            pass

if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!!!")