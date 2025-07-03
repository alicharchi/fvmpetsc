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

nx,ny = 10,20
Lx,Ly = 1,2
alpha = 1
bc = {'W':0,'E':0,'N':100,'S':0 }

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

command.append("./twoD")
command += ['-nx', str(nx)]
command += ['-ny', str(ny)]
command += ['-Lx', str(Lx)]
command += ['-Ly', str(Ly)]
command += ['-alpha', str(alpha)]
command += ['-bcW', str(bc['W'])]
command += ['-bcE', str(bc['E'])]
command += ['-bcN', str(bc['N'])]
command += ['-bcS', str(bc['S'])]

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
    dx,dy = Lx/nx, Ly/ny
    Ae = Aw = dy
    As = An = dx

    x = np.arange(dx/2,Lx,dx)
    y = np.arange(dy/2,Ly,dy)
            
    # generate 2 2d grids for the x & y bounds
    xm, ym = np.meshgrid(x, y)
    Tm = np.array(T).reshape(ny,nx)

    fig, ax = plt.subplots()
    cf = ax.contourf(xm,ym, Tm, cmap=plt.cm.jet, levels=200)
    fig.colorbar(cf, ax=ax)
    ax.set_title('Temperature')    
    fig.tight_layout()    
    fig.savefig('T.png')   
    plt.close(fig)   

else:
    print('Error Ocuured')
