import argparse
from gpu_helpers import print_details
from static_globals import *
from datasets import load_dataset
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from better_pipeline import BetterPipeline
import torch
import random
from distillation_helpers import reverse_step, clone_pipeline, default_lora_config
import torch.nn.functional as F
import time
from adapter_helpers import better_load_ip_adapter
from datetime import datetime
import os
import wandb
from memory_profiler import profile
import psutil
from experiment_helpers.measuring import get_metric_dict,METRIC_LIST
import numpy as np
from peft import get_peft_model

# getting the current date and time
current_datetime = datetime.now()

#torch.autograd.set_detect_anomaly(True)

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
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--use_negative_prompt",action="store_true")
parser.add_argument("--guidance_scale",type=float,default=7.5)
parser.add_argument("--shuffle",action="store_true")
parser.add_argument("--pretrain_noise_pipeline",action="store_true")
parser.add_argument("--use_ip_adapter",action="store_true")
parser.add_argument("--max_grad_norm",type=float,default=1.0)
parser.add_argument("--image_dir",type=str,default="/scratch/jlb638/distillation")
#TODO set sampler as arg
#TODO noise prediction vs x prediction
#TODO SNR coefficien
'''
Predicting x directly.

Predicting both x and epsilon, via separate output channels {x˜θ(zt), epsilon ˜θ(zt)} of the neural network, and then merging the predictions via 
xˆ = σ2t x˜θ(zt) + αt(zt − σt epsilon˜θ(zt)), thus
smoothly interpolating between predicting x directly and predicting via epsilon.


'''

@profile
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
        print("line 95 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        teacher_pipeline=BetterPipeline.from_pretrained(args.pretrained_path,safety_checker=None)
        print("line 97 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        teacher_pipeline("do this to help instantiate proerties",num_inference_steps=1)
        if args.use_ip_adapter:
            teacher_pipeline=better_load_ip_adapter(
                teacher_pipeline,"h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name,low_cpu_mem_usage=True
            )
            print("teacher pipeline loaded line 99")
            #teacher_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name)
            #teacher_pipeline("do this to help instantiate proerties",num_inference_steps=1,ip_adapter_image=image)
        '''else:
            teacher_pipeline("do this to help instantiate proerties",num_inference_steps=1)'''
        if args.use_lora:
            teacher_pipeline.unet=get_peft_model(teacher_pipeline.unet,default_lora_config)

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
        print("line 130 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        
        ip_adapter_image_embeds=None
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
        print("line 143 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        
        print("len prompt list",len(positive_prompt_list))
        print("len batched ",len(positive_prompt_list_batched))
        num_channels_latents = teacher_pipeline.unet.config.in_channels
        
        aggregate_dict={name: {metric:[] for metric in METRIC_LIST} for name in ["student","baseline","baseline_fast"]}
        if args.method_name==PROGRESSIVE:
            print("line 149 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        
            student_pipeline=clone_pipeline(args,teacher_pipeline,image)
            print("line 152 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        
            student_pipeline.scheduler.set_timesteps(args.initial_num_inference_steps)
            student_pipeline.unet=student_pipeline.unet.to(accelerator.device)
            print("line 156 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
            student_steps=args.initial_num_inference_steps//2
            accelerator.free_memory()
            torch.cuda.empty_cache()
            while student_steps>=args.final_num_inference_steps:
                accelerator.gradient_accumulation_steps=min(accelerator.gradient_accumulation_steps,student_steps )
                print("effective batch size ",accelerator.gradient_accumulation_steps * args.batch_size)
                print("line 163 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
                teacher_pipeline=student_pipeline
                teacher_pipeline.unet.requires_grad_(False)
                print("line 166 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
                student_pipeline=clone_pipeline(args,teacher_pipeline,image)
                student_pipeline.scheduler.set_timesteps(student_steps)
                student_pipeline.unet.requires_grad_(True)
                student_pipeline.unet=student_pipeline.unet.to(accelerator.device)
                print("student pipeline all ready")
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
                    print("begin epoch ",e)
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
                            steps=student_pipeline.scheduler.timesteps
                            for student_i in range(len(steps)):
                                with accelerator.accumulate(student_pipeline.unet):
                                    #print("prompt embeds size",prompt_embeds.size())

                                    start_latents=teacher_latents_plus.clone()

                                    student_t=steps[student_i]
                                    teacher_i=(2*student_i)-1
                                    teacher_t=student_t.clone() #teacher_pipeline.scheduler.timesteps[teacher_i]
            
                                    student_latents=reverse_step(args,student_t,student_pipeline,start_latents,prompt_embeds, added_cond_kwargs)
                                    
                                    teacher_latents=reverse_step(args,teacher_t, teacher_pipeline, start_latents, prompt_embeds, added_cond_kwargs)
                                    teacher_t_plus=teacher_pipeline.scheduler.timesteps[teacher_i+1]
                                    teacher_latents_plus=reverse_step(args,teacher_t_plus, teacher_pipeline, teacher_latents, prompt_embeds, added_cond_kwargs)
                                    print(student_t, teacher_t, teacher_t_plus)
                                    #compute loss
                                    
                                    loss=F.mse_loss(teacher_latents_plus,student_latents,reduction="mean")
                                    avg_loss+=loss.detach().cpu().numpy()/args.batch_size
                                    #print(loss.detach().cpu().numpy()/args.batch_size)
                                    accelerator.backward(loss,retain_graph=True)
                                    if accelerator.sync_gradients:
                                        accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
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
                    #validation images
                    save_dir=os.path.join(args.image_dir, "validation",f"steps_{student_steps}")
                    os.makedirs(save_dir, exist_ok=True)
                    #validation images
                    kwargs={
                        "prompt":subject,
                        "guidance_scale":1.0,
                        "num_inference_steps":student_steps
                    }
                    if args.do_classifier_free_guidance:
                        kwargs["guidance_scale"]=args.guidance_scale
                        kwargs["negative_prompt"]=negative_prompt
                    if args.use_ip_adapter:
                        kwargs["ip_adapter_image"]=image
                    validation_image=student_pipeline(**kwargs).images[0]
                    save_path=os.path.join(save_dir,f"_{e}.png")
                    validation_image.save(save_path)
                    try:
                        accelerator.log({
                            f"{student_steps}/{e}":wandb.Image(save_path)
                        })
                    except:
                        accelerator.log({
                            f"{student_steps}/{e}":validation_image
                        })
                    #check if epoch loss<convergence
                    if epoch_loss/(len(positive_prompt_list_batched))<args.convergence_threshold:
                        break

                student_steps=student_steps//2
                accelerator.free_memory()
                torch.cuda.empty_cache()
                #metrics
        elif args.method_name==CYCLE_GAN:
            noise_list=[]
            image_list=[]

            for prompt in training_prompt_list:
                noise_latents= teacher_pipeline.vae.config.scaling_factor * teacher_pipeline.prepare_latents(
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
            student_pipeline=clone_pipeline(args,teacher_pipeline,image)
            student_pipeline.unet=student_pipeline.unet.to(accelerator.device)
            print("line 305 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
            teacher_pipeline.scheduler.set_timesteps(args.initial_num_inference_steps)
            teacher_pipeline.unet=teacher_pipeline.unet.to(accelerator.device)
            student_pipeline.scheduler.set_timesteps(args.final_num_inference_steps)
            trainable_parameters=filter(lambda p: p.requires_grad, student_pipeline.unet.parameters())
            #print(trainable_parameters)
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=0.00000001)
            print("line 316 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
            for e in range(args.epochs):
                start=time.time()
                epoch_loss=0.0
                for positive,negative in zip(positive_prompt_list_batched, negative_prompt_list_batched):
                    #with accelerator.accumulate(student_pipeline.unet):
                    avg_loss=0.0
                    #TODO prepare and clone latents
                    start_latents = student_pipeline.vae.config.scaling_factor * student_pipeline.prepare_latents(
                        args.batch_size,
                        num_channels_latents,
                        args.size,
                        args.size,
                        positive.dtype,
                        accelerator.device,
                        generator)
                    start_latents = torch.cat([start_latents] * 2) if args.do_classifier_free_guidance else start_latents
                    #latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                    #teacher_latents=student_latents.clone()
                    #teacher_latents_plus=student_latents.clone()
                    positive=positive.to(accelerator.device)
                    #print("latennts size",start_latents.size())
                    
                    if args.do_classifier_free_guidance:
                        negative=negative.to(accelerator.device)
                        prompt_embeds = torch.cat([negative, positive])
                    else:
                        prompt_embeds=positive
                    #print("prompt_embeds size",prompt_embeds.size())
                    #print('start_latents device', start_latents.device)
                    #print('time device ',torch.tensor(1000,device=accelerator.device).device)
                    #print('student_pipeline.unet',student_pipeline.unet.device)
                    #print('prompt_embeds device',prompt_embeds.device)
                    student_noise_pred=student_pipeline.unet(
                            start_latents,
                            torch.tensor(1000,device=accelerator.device),
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=None,
                            cross_attention_kwargs=None,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                    )[0]
                    if args.do_classifier_free_guidance:
                        student_noise_pred_uncond, student_noise_pred_text = student_noise_pred.chunk(2)
                        student_noise_pred = student_noise_pred_uncond + args.guidance_scale * (student_noise_pred_text - student_noise_pred_uncond)
                    latents=start_latents.clone()
                    #print("inital latents size",latents.size())
                    steps=teacher_pipeline.scheduler.timesteps
                    for teacher_t in steps:
                        #print("inital latents size 335",latents.size())
                        with accelerator.accumulate(student_pipeline.unet):
                            #print("inital latents size 337",latents.size())
                            latent_model_input = latents
                            #print("inital latents size 339",latents.size())
                            latent_model_input = teacher_pipeline.scheduler.scale_model_input(latent_model_input, teacher_t)
                            #print("latent_model_input size",latent_model_input.size())
                            print('teacher_pipeline.unet',teacher_pipeline.unet.device)
                            print("atent_model_input",latent_model_input.device)
                            print("teacher_t",teacher_t.device)
                            print("prompt_embeds",prompt_embeds.device)
                            noise_pred = teacher_pipeline.unet(
                                latent_model_input,
                                teacher_t.to(accelerator.device),
                                encoder_hidden_states=prompt_embeds,
                                timestep_cond=None,
                                cross_attention_kwargs=None,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0]
                            latents_pred=latent_model_input-noise_pred
                            #print("noise pred size", noise_pred.size())    
                            '''if args.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                                print('cfg noise pred size',noise_pred.size())'''
                            student_noise_pred=student_pipeline.unet(
                                    start_latents,
                                    torch.tensor(1000,device=accelerator.device),
                                    encoder_hidden_states=prompt_embeds,
                                    timestep_cond=None,
                                    cross_attention_kwargs=None,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                            )[0]
                            #print("student_noise_pred size",student_noise_pred.size())
                            student_latents_pred=start_latents-student_noise_pred
                            '''if args.do_classifier_free_guidance:
                                student_noise_pred_uncond, student_noise_pred_text = student_noise_pred.chunk(2)
                                student_noise_pred = student_noise_pred_uncond + args.guidance_scale * (student_noise_pred_text - student_noise_pred_uncond)
                                print('student_noise_pred cfg ', student_noise_pred.size())'''
                            
                            loss=F.mse_loss(latents_pred,student_latents_pred,reduction="mean")
                            avg_loss+=loss.detach().cpu().numpy()/effective_batch_size
                            #print('avg_loss',loss.detach().cpu().numpy()/effective_batch_size)
                            #print("line 404 psutil", psutil.cpu_percent(),psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
                            accelerator.backward(loss,retain_graph=True)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()

                            latents = teacher_pipeline.scheduler.step(noise_pred, teacher_t, latents, return_dict=False)[0]
                            #latents = torch.cat([latents] * 2) if args.do_classifier_free_guidance else latents
                            print('latents size',latents.size())
                print("epoch avg loss", avg_loss)
                end=time.time()
                print(f"epochs {e} ended after {end-start} seconds = {(end-start)/3600} hours")
                inference_step_list=[inference_steps for inference_steps in range(1,args.final_num_inference_steps)]
                save_dir=os.path.join(args.image_dir, "validation",f"epoch_{e}")
                os.makedirs(save_dir, exist_ok=True)
                for inference_steps in inference_step_list:
                    #validation images
                    kwargs={
                        "prompt":subject,
                        "guidance_scale":1.0,
                        "num_inference_steps":inference_steps,
                        "height":args.size,
                        "width":args.size
                    }
                    if args.do_classifier_free_guidance:
                        kwargs["guidance_scale"]=args.guidance_scale
                        kwargs["negative_prompt"]=negative_prompt
                    if args.use_ip_adapter:
                        kwargs["ip_adapter_image"]=image
                    validation_image=student_pipeline(**kwargs).images[0]
                    save_path=os.path.join(save_dir,f"_{inference_steps}.png")
                    validation_image.save(save_path)
                    try:
                        accelerator.log({
                            f"{e}/{inference_steps}":wandb.Image(save_path)
                        })
                    except:
                        accelerator.log({
                            f"{e}/{inference_steps}":validation_image
                        })
                accelerator.free_memory()
                torch.cuda.empty_cache()
        accelerator.free_memory()
        torch.cuda.empty_cache()
        eval_prompt_list=[
        "a photo of  {} at the beach",
        "a photo of  {} in the jungle",
        "a photo of  {} in the snow",
        "a photo of  {} in the street",
        "a photo of  {} with a city in the background",
        "a photo of  {} with a mountain in the background",
        "a photo of  {} with the Eiffel Tower in the background",
        "a photo of  {} near the Statue of Liberty",
        "a photo of  {} near the Sydney Opera House",
        "a photo of  {} floating on top of water",
        "a photo of  {} eating a burger",
        "a photo of  {} drinking a beer",
        "a photo of  {} wearing a blue hat",
        "a photo of  {} wearing sunglasses",
        "a photo of  {} playing with a ball",
        "a photo of  {} as a police officer"]
        baseline_pipeline=BetterPipeline.from_pretrained(args.pretrained_path,safety_checker=None)
        for pipe in [baseline_pipeline, student_pipeline]:
            baseline_pipeline("instantiate things",num_inference_steps=1)
            baseline_pipeline.safety_checker=None

        ip_adapter_image_embeds_cpu=None
        ip_adapter_image_embeds_device=None
        if args.use_ip_adapter:
            baseline_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=args.ip_weight_name,low_cpu_mem_usage=True)
            baseline_pipeline.image_encoder=baseline_pipeline.image_encoder.to(accelerator.device)
            baseline_pipeline.image_encoder.eval()
            student_pipeline.image_encoder=student_pipeline.image_encoder.to("cpu")
            student_pipeline.image_encoder.eval()
            ip_adapter_image_embeds_cpu=[torch.stack([ip_adapter_image_embeds.to("cpu") for _ in range(2)])]
            ip_adapter_image_embeds_device=[torch.stack([ip_adapter_image_embeds.to(accelerator.device) for _ in range(2)])]
        accelerator.free_memory()
        torch.cuda.empty_cache()
        baseline_pipeline.unet=baseline_pipeline.unet.to(accelerator.device)
        baseline_pipeline.text_encoder=baseline_pipeline.text_encoder.to(accelerator.device)
        baseline_pipeline.vae=baseline_pipeline.vae.to(accelerator.device)

        student_pipeline.text_encoder=student_pipeline.text_encoder.to("cpu")
        student_pipeline.unet=student_pipeline.unet.to("cpu")
        student_pipeline.vae=student_pipeline.vae.to("cpu")

        for model in [baseline_pipeline.unet, baseline_pipeline.text_encoder,baseline_pipeline.vae,student_pipeline.text_encoder,student_pipeline.unet ,student_pipeline.vae]:
            model.eval()
            print(model.device)
        student_image_list=[student_pipeline(prompt=prompt.format(subject), num_inference_steps=args.final_num_inference_steps, ip_adapter_image_embeds=ip_adapter_image_embeds_cpu,safety_checker=None,height=args.size,width=args.size).images[0] for prompt in eval_prompt_list]
        baseline_image_list=[baseline_pipeline(prompt=prompt.format(subject), num_inference_steps=args.initial_num_inference_steps, ip_adapter_image_embeds=ip_adapter_image_embeds_device,safety_checker=None,height=args.size,width=args.size).images[0] for prompt in eval_prompt_list]
        fast_baseline_list=[baseline_pipeline(prompt=prompt.format(subject), num_inference_steps=args.final_num_inference_steps, ip_adapter_image_embeds=ip_adapter_image_embeds_device,safety_checker=None,height=args.size,width=args.size).images[0] for prompt in eval_prompt_list]
        for name,image_list in zip(["student","baseline","baseline_fast"],[student_image_list, baseline_image_list, fast_baseline_list]):
            metric_dict=get_metric_dict([prompt.format(subject) for prompt in eval_prompt_list], image_list, [image])
            for metric,value in metric_dict.items():
                aggregate_dict[name][metric].append(value)
                print(f"\t{metric} {value}")
    for name,aggregates in aggregate_dict.items():
        for metric,value_list in aggregates.items():
            print(f"\t{metric} {np.mean(value_list)}")
            accelerator.log({
                f"{name}_"+metric:np.mean(value_list)
            })



if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    start=time.time()
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful training :) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!!!")