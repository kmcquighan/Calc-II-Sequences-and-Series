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
This function considers the continuous function case with a small fixed right endpoint.           
"""
def exploreContinuous(f,b):
    
    a = 1.
    xmax = 10
    func = eval("lambda x: " + f)
        
    fig = plt.figure(figsize=(20, 6))       
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    x = np.linspace(.5,11,1000)
    idx_a = np.where(x>=a)[0][0]
    idx_b = np.where(x>b)[0][0]
    y = func(x)
    ax1.set_xlim([0,1.03*xmax])
    ax1.set_ylim([0,func(0.5)])
    ax2.set_xlim([0,1.03*xmax])
    ax2.set_ylim([0,inte.quad(func,1.,11.)[0]])
    if min(y)>0.:
        ax1.set_ylim([0., max(y)])
    elif max(y)<0.:
        ax1.set_ylim([min(y), 0.])
    else:
        ax1.set_ylim([1.1*min(y), 1.1*max(y)])
    
    ax1.plot(x,y,'b',linewidth=5)
    ax1.fill_between(x[idx_a:idx_b],y[idx_a:idx_b], facecolor='g', edgecolor='g', alpha=0.3, linewidth=3)
    I = np.zeros(1000)
    for i in range(1000):
        I[i] = inte.quad(func,1.,x[i])[0]
    ax2.plot(x[idx_a:idx_b],I[idx_a:idx_b],'r',linewidth=5)
    ax2.plot(b,I[idx_b-1],'go',markersize=13)
    if min(I[idx_a:])>0.:
        ax2.set_ylim([0., 1.1*max(I)])
    elif max(I[idx_a:])<0.:
        ax2.set_ylim([1.1*min(I), 0.])
    else:
        ax2.set_ylim([1.1*min(I), 1.1*max(I)])
    
    ax1.set_xlabel('x', fontsize=36)
    ax1.set_title('f(x)', fontsize=36)
    ax2.set_xlabel('b', fontsize=36)
    ax2.set_title(r'$\int_1^b f(x)dx$',fontsize=36, y=1.1)
    
    ax1.axhline(0.,color='k',linewidth=1)
    ax2.axhline(0.,color='k',linewidth=1)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.suptitle('f(x) = '+f, fontsize=36, y=1.0)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)

"""
This function considers the discrete (sequence) case with a small fixed right endpoint.
""" 
def exploreDiscrete(f,n):     

    n = int(n)
    func = eval("lambda n: " + f)
    nmax = int(10)
    
    fig = plt.figure(figsize=(20, 6))
       
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
      
    ns = np.linspace(1,nmax,nmax)
    an = func(ns)
    ax1.set_xlim([0,1.03*nmax])
    ax2.set_xlim([0,1.03*nmax])
    
    ax1.plot(ns,an,'bo',markersize=13)
    sn = np.zeros(nmax)
    sn[0] = an[0]
    ax2.axhline(sn[0],color='k',linestyle=':', linewidth=1)
    hl = min([0.1, 0.6*abs(an[0])])
    ax1.arrow(1, 0, 0, an[0]-np.sign(an[0])*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc='g', ec='g')
    ax2.arrow(n-0.2, 0, 0, sn[0]-np.sign(an[0])*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc='g', ec='g')
    
    for i in range(1,n):
        hl = min([0.1, 0.6*abs(an[i])])
        sn[i] = sn[i-1]+an[i]
        ax2.axhline(sn[i],color='k',linestyle=':', linewidth=1)
        ax1.arrow(i+1, 0, 0, an[i]-np.sign(an[i])*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc='g', ec='g')
        ax2.arrow(n-.2, sn[i-1], 0, an[i]-np.sign(an[i])*hl, head_width=0.1, head_length=abs(hl),linewidth=5, fc='g', ec='g')
    
    for i in range(n,nmax):
        sn[i] = sn[i-1]+an[i]
    
    hl = np.min(0.1, 0.6*an[0])
    
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
This function considers the continuous (function) case where the right endpoint 
increases exponentially.
"""    
def plotContinuous(f,m):
    
    a = 1.
    b = 10**m
    func = eval("lambda x: " + f)
        
    fig = plt.figure(figsize=(20, 6))
       
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    x = np.linspace(a,b,1000)
    x_all = np.linspace(0.5,1.03*b,1000)
    y = func(x)
    y_all = func(x_all)
    ax1.set_xlim([0,1.03*b])
    ax2.set_xlim([0,1.03*b])
    if min(y_all)>0.:
        ax1.set_ylim([0., max(y_all)])
    elif max(y_all)<0.:
        ax1.set_ylim([min(y_all), 0.])
    else:
        ax1.set_ylim([min(y_all), max(y_all)])
    
    ax1.plot(x_all,y_all,'b',linewidth=5)
    ax1.fill_between(x,y, facecolor='g', edgecolor='g', alpha=0.3, linewidth=3)
    I = np.zeros(1000)
    for i in range(1000):
        I[i] = inte.quad(func,1.,x[i])[0]
    
    if min(I)>0.:
        ax2.set_ylim([0., 1.1*max(I)])
    elif max(I)<0.:
        ax2.set_ylim([1.1*min(I), 0.])
    else:
        ax2.set_ylim([1.1*min(I), 1.1*max(I)])
    ax2.plot(x,I,'r',linewidth=5)
    ax2.plot(x[999], I[999],'go', markersize=13)
    
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

"""
This function considers the discrete (sequence) case where the right endpoint 
increases on an exponential scale.
""" 
def plotDiscrete(f,m):     

    b = int(10**m)
    func = eval("lambda n: " + f)
        
    fig = plt.figure(figsize=(20, 6))
       
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
      
    n = np.linspace(1,b,b)
    an = func(n)
    ax1.set_xlim([0,1.03*b])
    ax2.set_xlim([0,1.03*b])
    
    sn = np.zeros(b)
    sn[0] = an[0]
    ones = np.ones(2)
    ax1.plot(ones,[0.,an[0]],color='g',linewidth=5)
    for i in range(b-1):
        sn[i+1] = sn[i]+an[i+1]
        ax1.plot((i+2)*ones,[0.,an[i+1]],color='g',linewidth=5)
    ax1.plot(n,an,'bo',markersize=8)
    ax2.plot(n,sn,'ro',markersize=8)
    ax2.plot(b,sn[b-1],'go',markersize=13)
    
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
