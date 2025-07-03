#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nCores", type=int, help="Number of parallel cores")
args = parser.parse_args()

n = 10
L = 1
alpha = 1
bc = {'W':0,'E':100}

nCores = 1
if (args.nCores) and (args.nCores!=1):
    nCores = args.nCores
    print(f"Attempting to run with {nCores} parallel cores")
else:
    print(f"Attempting to run in serial mode")

outPath = 'T.out'

if os.path.isfile(outPath):
    os.remove(outPath)

command = []
if (nCores>1):
    command.append("mpiexec")
    command += ["-n" ,str(nCores)]

command.append("./oneD")
command += ['-nx', str(n)]
command += ['-L', str(L)]
command += ['-alpha', str(alpha)]
command += ['-bcW', str(bc['W'])]
command += ['-bcE', str(bc['E'])]

r = subprocess.run(command)

if (r.returncode==0): #Success
    T = []

    with open(outPath, 'r') as file1:    
        while True:     
            
            line = file1.readline()
            
            if not line:
                break

            a = line.strip()

            res = re.match('^[0-9\.]',a)        
            if (res is not None):
                T.append(float(a))

    #grid generation
    dx = L/n
    Ae = Aw = 1
    
    # generate grid
    x = np.arange(dx/2,L,dx)               
    Tm = np.array(T)

    fig, ax = plt.subplots()
    ax.plot(x,Tm)    
    ax.set_title('Temperature')    
    fig.tight_layout()    
    fig.savefig('T.png')   
    plt.close(fig)   

else:
    print('Error Ocuured')
