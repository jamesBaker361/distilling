import argparse
from gpu_helpers import print_details

parser=argparse.ArgumentParser()

def main(args):
    return

if __name__=='__main__':
    print_details()
    args=parser.parse_args()
    print(args)
    main(args)
    print("all done!!!")