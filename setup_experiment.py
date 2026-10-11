"""Build native kernels for the small-instance experiment."""
from pathlib import Path
import argparse
import subprocess
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cuda',action='store_true',help='Also compile the CUDA kernels with nvcc')
    args=parser.parse_args()
    for source,target in [('bellman.cpp','bellman.so'),('bellman_compact_cpu.cpp','bellman_compact_cpu.so')]:
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/source),'-o',str(ROOT/target)],check=True)
    if args.cuda:
        for source,target in [('bellman_compact.cu','bellman_compact.so'),('bellman_gpu.cu','bellman_gpu.so')]:
            subprocess.run(['nvcc','-O3','-std=c++17','--shared','-Xcompiler','-fPIC',str(ROOT/source),'-o',str(ROOT/target)],check=True)
    results=ROOT/'results';results.mkdir(exist_ok=True)
    print('Setup complete.')

if __name__=='__main__':main()
