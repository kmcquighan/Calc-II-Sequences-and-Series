# -*- coding: utf-8 -*-
"""
Kelly McQuighan 2017.

These tools can be used to visualize the concept of convergence of an infinite sequence
versus an infinite series. An analogy is made with the convergence of a function and
the convergence of its integral.

Both the continuous (function) and the discrete (sequences) cases have two methods. 
The 'explore' method has a fixed maximum value of the right end point. The purpose
of this restriction is to help the use to first understand how the plots are generated. 
The 'plot' method has a much larger right endpoint and in fact is placed on a log-scale
to make it easier for the user to increase the right endpoint quickly.
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from numpy import *
import scipy.integrate as inte
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['font.size'] = 20

"""
This function plots the continuous case based on the choice for right endpoint.
It is a helper function that is called by both smallContinuous and largeContinuous
where the only difference is how the right endpoint is scaled.
"""
def plotContinuous(f,b,xmax):
    a = 1.
    func = eval("lambda x: " + f)
        
    fig = plt.figure(figsize=(20, 6))
       
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    x = np.linspace(0.5,xmax,1000)
    y = func(x)
    start = np.where(x>=a)[0][0]
    end = np.where(x>b)[0][0]
    ax1.set_xlim([0,xmax])
    ax2.set_xlim([0,xmax])
    if min(y)>0.:
        ax1.set_ylim([0., max(y)])
    elif max(y)<0.:
        ax1.set_ylim([min(y), 0.])
    else:
        ax1.set_ylim([min(y), max(y)])
    
    ax1.plot(x,y,'b',linewidth=5)
    ax1.fill_between(x[start:end],y[start:end], facecolor='g', edgecolor='g', alpha=0.3, linewidth=3)
    I = np.zeros(1000)
    for i in range(1000):
        I[i] = inte.quad(func,1.,x[i])[0]
    
    if min(I[start:])>0.:
        ax2.set_ylim([0., max(I[start:])])
    elif max(I[start:])<0.:
        ax2.set_ylim([min(I[start:]), 0.])
    else:
        ax2.set_ylim([min(I[start:]), max(I[start:])])
    
    ax2.plot(x[start:end],I[start:end],'r',linewidth=5)
    ax2.plot(x[end], I[end],'go', markersize=13)
    
    ax1.axhline(0.,color='k',linewidth=1)
    ax2.axhline(0.,color='k',linewidth=1)
    
    ax1.set_xlabel('x', fontsize=36)
    ax1.set_title('f(x)', fontsize=36)
    ax2.set_xlabel('b', fontsize=36)
    ax2.set_title(r'$\int_1^b f(x)dx$',fontsize=36, y=1.1)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.suptitle('f(x) = '+f+ ' for $x\in$ [1, %.2e]' %b, fontsize=36, y=1.2)
    plt.suptitle('f(x) = '+f, fontsize=36, y=1.0)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)

# helper function to plot the arrow head correctly
def plot_arrow(ani,sni,sn_iMinus1,i,n,ax1,ax2):
    hl = min([0.1, 0.6*abs(ani)])
    ax2.axhline(sni,color='k',linestyle=':', linewidth=1)
    ax1.arrow(i+1, 0, 0, ani-np.sign(ani)*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc='g', ec='g')
    ax2.arrow(n-.2, sn_iMinus1, 0, ani-np.sign(ani)*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc='g', ec='g')
    

"""
This function plots the discrete case based on the choice for right endpoint.
It is a helper function that is called by both smallDiscrete and largeDiscrete
where the only difference is how the right endpoint is scaled.
"""   
def plotDiscrete(f,n,nmax,show_arrow):
    
    n = int(n)
    func = eval("lambda n: " + f)
    
    fig = plt.figure(figsize=(20, 6))
       
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
      
    ns = np.linspace(1,nmax,nmax)
    an = func(ns)
    ax1.set_xlim([0,1.03*nmax])
    ax2.set_xlim([0,1.03*nmax])
    
    sn = np.zeros(nmax)
    sn[0] = an[0]

    ones = np.ones(2)
    if (show_arrow): plot_arrow(an[0],sn[0],0,0,n,ax1,ax2)
    else: ax1.plot(ones,[0.,an[0]],color='g',linewidth=5) #plots a line
        
    for i in range(1,n):
        sn[i] = sn[i-1]+an[i]

        if (show_arrow): plot_arrow(an[i],sn[i],sn[i-1],i,n,ax1,ax2)
        else: ax1.plot((i+1)*ones,[0.,an[i]],color='g',linewidth=5)
            
    for i in range(n,nmax):
        sn[i] = sn[i-1]+an[i]
    
    ax1.plot(ns,an,'bo',markersize=13)
    
    if min(an)>0.:
        ax1.set_ylim([0., 1.1*max(an)])
    elif max(an)<0.:
        ax1.set_ylim([1.1*min(an), 0.])
    else:
        ax1.set_ylim([1.1*min(an), 1.1*max(an)])

    if min(sn)>0.:
        ax2.set_ylim([0., 1.1*max(sn)])
    elif max(sn)<0.:
        ax2.set_ylim([1.1*min(sn), 0.])
    else:
        ax2.set_ylim([1.1*min(sn)-0.1, 1.1*max(sn)+0.1])

    ax2.plot(ns[:n],sn[:n],'ro',markersize=10)
    ax2.plot(n,sn[n-1],'go', markersize=13)
    
    ax1.axhline(0.,color='k',linewidth=1)
    ax2.axhline(0.,color='k',linewidth=1)
    ax1.set_xlabel('n', fontsize=36)
    ax1.set_title(r'$a_n$', fontsize=36, y=1.1)
    ax2.set_xlabel('k', fontsize=36)
    ax2.set_title(r'$s_k=\sum_{n=1}^k a_n$',fontsize=36, y=1.1)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.suptitle(r'$a_n$ = '+f, fontsize=36, y=1.0)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)

"""
This function considers the continuous function case with a small fixed right endpoint.           
"""    
def smallContinuous(f,b):
    
    plotContinuous(f,b,11)
    

"""
This function considers the continuous (function) case where the right endpoint 
increases exponentially.
"""    
def largeContinuous(f,m):
    
    plotContinuous(f,10**m,1.05*10**m)
    

"""
This function considers the discrete (sequence) case with a small fixed right endpoint.
""" 
def smallDiscrete(f,n):     
    
    plotDiscrete(f,n,11,True)


"""
This function considers the discrete (sequence) case where the right endpoint 
increases on an exponential scale.
""" 
def largeDiscrete(f,m):     

    plotDiscrete(f,10**m,int(1.05*10**m),False)
