import argparse
from gpu_helpers import print_details
from static_globals import *
from datasets import load_dataset
from accelerate import Accelerator

parser=argparse.ArgumentParser()

parser.add_argument("--method_name",type=str,default=PROGRESSIVE)
parser.add_argument("--dataset",type=str,default="jlbaker361/league-hard-prompt")
parser.add_argument("--image_field_name",type=str,default="tile")
parser.add_argument("--text_field_name",type=str,default="subject")
parser.add_argument("--limit",type=int,default=10)
parser.add_argument("--gradient_accumulation_steps",default=8,type=int)
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="distillation")

def main(args):
    data=load_dataset(args.dataset,split="train")
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    for i, row in enumerate(data):
        if i>args.limit:
            break
        if args.method_name==PROGRESSIVE:
            return

if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!!!")