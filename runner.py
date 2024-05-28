import argparse
from gpu_helpers import print_details
from static_globals import *

parser=argparse.ArgumentParser()

parser.add_argument("--method_name",type=str,default=PROGRESSIVE)

def main(args):
    return

if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!!!")