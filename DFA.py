#!/usr/bin/env python3

import sys
import bz2
import numpy as np

import scipy.stats as st
import math

import matplotlib.pyplot as plt

# MATPLOTLIB
plt.rc('patch',linewidth=1)
plt.rc('axes', linewidth=1, labelpad=2)
plt.rc('xtick.minor', size=2, width=1)
plt.rc('xtick.major', size=4, width=1, pad=2)
plt.rc('ytick.minor', size=2, width=1)
plt.rc('ytick.major', size=4, width=1, pad=2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Computer Modern', size=6)
# MATPLOTLIB

def dfa(Y, nbins):
    L = len(Y) # L>20
        
    lmin = 10
    lmax = int(L/2)
    logdelta=(math.log10(lmax)-math.log10(lmin))/nbins
    
    ell = []
    F = []
    for k in range(nbins):
        l = int(math.pow(10,math.log10(lmin)+k*logdelta))
        
        n = int(L/l) # Nonoverlapping boxes
        
        F2 = 0.
        for i in range(n):
            if(i==n-1):
                x = list(range(i*l,L))
                y = Y[i*l:L]
            else:
                x = list(range(i*l, (i+1)*l))
                y = Y[i*l:(i+1)*l]
            
            a, b, rval, pval, err = st.linregress(x, y)
            yl = list(map(lambda X: a*X+b, x)) # Local trend
            
            f2 = 0.
            for j in range(l):
                f2 = f2+(y[j]-yl[j])**2 # Detrended walk
            f2 = f2/l # Squared fluctuation
            
            F2 = F2+f2
        F2 = F2/n # Average of the squared fluctuations
        
        ell.append(l)
        F.append(math.sqrt(F2))
    
    return ell, F

filename0=sys.argv[1]
filename1=sys.argv[2]
filename2=sys.argv[3]

with bz2.open('../datasets/'+filename0+'/'+filename1+'/'+filename2+'.bz2', 'rt', encoding='utf-8') as f:
    line=f.readline()
    field=line.split()
    lx=int(field[0])
    ly=int(field[1])
#    lon=float(field[2])
#    lat=float(field[3])
#    delta=float(field[4])
#    R=float(field[5])
    height=np.zeros((ly,lx))
    
    for line in f:
        field=line.split()
        
        i=int(field[0])
        j=int(field[1])
        
        height[j,i]=float(field[2])

alpha=[]
# VERTICAL
for j in range(ly):
    y=[]
    for i in range(lx):
        y.append(height[j,i])
    
    ell,F=dfa(y,25)
    
    log10ell=list(map(lambda X: math.log10(X),ell))
    log10F=list(map(lambda X: math.log10(X),F))
    
    a,b,rval,pval,err=st.linregress(log10ell,log10F)
    alpha.append(a)
# VERTICAL

# HORIZONTAL
for i in range(lx):
    y=[]
    for j in range(ly):
        y.append(height[j,i])
    
    ell,F=dfa(y,25)
    
    log10ell=list(map(lambda X: math.log10(X),ell))
    log10F=list(map(lambda X: math.log10(X),F))
    
    a,b,rval,pval,err=st.linregress(log10ell,log10F)
    alpha.append(a)
# HORIZONTAL

# PLOT
figx=7 #3.42, 4.5, 7
figy=(7/10)*figx

fig,ax=plt.subplots(figsize=(figx,figy),dpi=600)

n,bins,patches=ax.hist(alpha,bins='fd', density=True, facecolor='#D55E00', alpha=0.75)

mu=sum(alpha)/(lx+ly)
mu_str='{:.2f}'.format(mu)

annotate=r'\begin{center}$\mu='+mu_str+'$\end{center}'
ax.annotate(annotate,xy=(0.95, 0.95),xycoords='axes fraction',ha='right',va='top',fontsize=12)

ax.axvline(x=mu,color='#000000',linestyle='--', linewidth=1)

ax.set_xlabel(r'$H$',fontsize=16)
ax.set_ylabel(r'\#',fontsize=16)

plt.savefig('hist.pdf', bbox_inches='tight')
# PLOT
