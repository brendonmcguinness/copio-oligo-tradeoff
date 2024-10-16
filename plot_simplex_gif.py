#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:37:11 2023

@author: brendonmcguinness
"""
#plot simplex for FD and dist from centroid figures
import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt

#weird warning with seaborn
import seaborn as sns
import imageio
import os
import pandas as pd
import matplotlib.lines as mlines

from shapely.geometry import Polygon
from scipy.integrate import odeint
#from scipy import stats
#from lag_v_budg_fn import get_rank_dist_save_ind
from lag_v_budg_fn import model
from lag_v_budg_fn import centeroidnp
from lag_v_budg_fn import get_fd_and_centroid
from lag_v_budg_fn import bary2cart
from lag_v_budg_fn import pick_inout_hull
from scipy.spatial import ConvexHull
from lag_v_budg_fn import get_rank_dist_save_ind
from lag_v_budg_fn import simplex_vertices
from lag_v_budg_fn import avg_eq_time_traits
from lag_v_budg_fn import weighted_centroid
from matplotlib import cm

def plot_simplex_1frame(a_eq,s,ts):
    #make simplex
    S,R = a_eq.shape
    plasma = cm.get_cmap('tab10', 10)
    #ind_del = 4
    #get barycentric coordinates
    ac,corners = bary2cart(a_eq,corners=simplex_vertices(R-1))
    
    hull = ConvexHull(ac)
    hull_total = ConvexHull(corners)
    #cent = weighted_centroid(ac,n)

    plt.figure()
    #plt.plot(ac[:,0], ac[:,1], 'o')
    cmaps = np.arange(S)
    #plt.scatter(cent[0],cent[1],s=200,marker='d',color='gold',zorder=2)

    plt.scatter(ac[:,0], ac[:,1], c=cmaps, cmap='tab10',s=50)
    for simplex in hull.simplices:
        plt.plot(ac[simplex, 0], ac[simplex, 1], 'k',linestyle='dashed',alpha=0.01) #linestyle='dashed'
    plt.fill(ac[hull.vertices,0], ac[hull.vertices,1], 'k-', alpha=0.1)    
    
    
    sc = bary2cart(s)[0]
    plt.scatter(sc[0],sc[1],s=350,marker='*',color='k',alpha=0.8)
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
    
    """
    ind=0
    for i in range(0,S):
        if i==ind:
            continue
        else:
            plt.plot((ac[ind,0],ac[i,0]),(ac[ind,1],ac[i,1]),color='k',linestyle='dotted',alpha=0.7)
    plt.plot((ac[ind,0],sc[0]),(ac[ind,1],sc[1]),color='r',linestyle='dashed',alpha=1)
"""
    plt.xlim(0,1)
    plt.axis('off')
    #plt.text(0.8,0.8,'t='+str(ts),fontsize=20)
    
def plot_simplex_weighted_1frame(a_eq,s,ts,n):
    #make simplex
    S,R = a_eq.shape
    plasma = cm.get_cmap('tab10', 10)
    #ind_del = 4
    #get barycentric coordinates
    ac,corners = bary2cart(a_eq,corners=simplex_vertices(R-1))
    
    hull = ConvexHull(ac)
    hull_total = ConvexHull(corners)
    sizes = 600*n/n.sum()
    sizes[sizes<15] = 15
    cent = weighted_centroid(ac,n)

    plt.figure()
    #plt.plot(ac[:,0], ac[:,1], 'o')
    cmaps = np.arange(S)
    plt.scatter(ac[:,0], ac[:,1],s=sizes, c=cmaps, cmap='tab10')
    for simplex in hull.simplices:
        plt.plot(ac[simplex, 0], ac[simplex, 1], 'k',linestyle='dashed',alpha=0.01) #linestyle='dashed'
    plt.fill(ac[hull.vertices,0], ac[hull.vertices,1], 'k-', alpha=0.1)    
    plt.scatter(cent[0],cent[1],s=250,marker='d',color='gold',zorder=2)

    
    sc = bary2cart(s,corners=simplex_vertices(R-1))[0]
    plt.scatter(sc[0],sc[1],s=350,marker='*',color='k',alpha=0.8)
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
    
    """
    ind=0
    for i in range(0,S):
        if i==ind:
            continue
        else:
            plt.plot((ac[ind,0],ac[i,0]),(ac[ind,1],ac[i,1]),color='k',linestyle='dotted',alpha=0.7)
    plt.plot((ac[ind,0],sc[0]),(ac[ind,1],sc[1]),color='r',linestyle='dashed',alpha=1)
"""
    #plt.xlim(0,1)
    #plt.axis('off')
    black_star = mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                          markersize=10, label='supply vector')
    gold_diamond = mlines.Line2D([], [], color='gold', marker='d', linestyle='None',
                          markersize=10, label='centroid')
    traits = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None',
                          markersize=10, label='traits')
    plt.legend(handles=[traits, black_star, gold_diamond],fontsize=12)


    plt.xlim(0,1)
    plt.axis('off')
    plt.text(0.4, 0.78, 'i=1', fontsize=16, fontstyle='italic')
    plt.text(-0.06, -0.06, 'i=2', fontsize=16, fontstyle='italic')
    plt.text(0.86, -0.06, 'i=3', fontsize=16, fontstyle='italic')

    #plt.text(0.8,0.8,'t='+str(ts),fontsize=20)

def plot_simplex_1frame_supplyline(a_eq,s,ts,cent):
    #make simplex
    S,R = a_eq.shape
    plasma = cm.get_cmap('tab10', 10)
    #ind_del = 4
    #get barycentric coordinates
    ac,corners = bary2cart(a_eq)
    
    hull = ConvexHull(ac)
    hull_total = ConvexHull(corners)
    #cent = get_fd_and_centroid(a_eq)[1]

    plt.figure()
    #plt.plot(ac[:,0], ac[:,1], 'o')
    cmaps = np.arange(S)
    plt.scatter(ac[:,0], ac[:,1], c=cmaps, cmap='tab10',s=50)
    for simplex in hull.simplices:
        plt.plot(ac[simplex, 0], ac[simplex, 1], 'k',linestyle='dashed',alpha=0.01) #linestyle='dashed'
    plt.fill(ac[hull.vertices,0], ac[hull.vertices,1], 'k-', alpha=0.1)    
    #plt.scatter(cent[0],cent[1],s=200,marker='d',color='k',alpha=0.4)

    
    sc = bary2cart(s,corners=simplex_vertices(R-1))[0]
    plt.scatter(sc[0],sc[1],s=350,marker='*',color='k',alpha=0.8)
    for simplex in hull_total.simplices:
        plt.plot(corners[simplex, 0], corners[simplex, 1], 'k-')
    plt.plot((ac[3,0],sc[0]),(ac[3,1],sc[1]),color='k',linestyle='dashed',alpha=0.7)

    
    plt.xlim(0,1)
    plt.axis('off')
    #plt.text(0.8,0.8,'t='+str(ts),fontsize=20)    
    
def plot_gif(a,n,t,s,num_frames,filenamegif='3species_weighted_070824.gif'):
    
    num_t = a.shape[0]
    #eq_t = avg_eq_time_traits(a, t)
    #print(eq_t)
    a0=a[0,:,:]
    cent = get_fd_and_centroid(a0)[1]
    
    #5.5
    t_plot = np.logspace(1,5.5,num_frames,dtype=int)
    t_plot = np.append(t_plot,num_t-1)
    #print(t_plot)
    filenames = []
    for i,ts in enumerate(t_plot-1):
        #print(ts)
        # plot the line chart
        #plot_simplex_1frame(a[ts+1,:,:],s,t_plot[i])  
        plot_simplex_weighted_1frame(a[ts+1,:,:],s,t_plot[i],n[ts+1,:]) 
        #plot_simplex_1frame_supplyline(a[ts+1,:,:],s,t_plot[i],cent)        

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()# build gif
    with imageio.get_writer(filenamegif, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    
if __name__ == "__main__":       
    S=10
    R=3
    v = np.random.uniform(10e7, 10e7, R)
    dlta = np.random.uniform(5e-3, 5e-3, S) 
    Q = np.ones(S)*10e-5 #random.uniform(10e-5,10e-5)
    E0 = np.random.uniform(Q*dlta, Q*dlta, S)
    K = np.random.uniform(10e-6, 10e-6, R)  # 10e-4 * np.ones(R)
    u = np.zeros(R)
    #s = np.array([7e-2,10e-2,1e-2,4e-3,8e-3,1e-2,6e-2,7e-3]) #np.random.uniform(10e-4, 10e-2, R)  # 100*np.zeros(R)
    s = np.random.uniform(10e-4,10e-2,R) #back to 10e-5
    #s = np.array([10e-2,5e-3,10e-3])
    #try with and without adaptive strategies
    d = 10e-6*np.ones(S) #10e-6*np.ones(S) * 0
    #d = np.array([10e-6,0,0,0,0,0,0,0,0,0])
    # time points
    num_t = 800000
    t_end = 40000 
    t = np.linspace(0, t_end, num_t)
    dira = np.array([2.0,4.0,8.0])
    
    a0 = np.zeros((S, R), dtype=float)
    for i in range(0, S):
        a0[i, :] = np.random.dirichlet(2*np.ones(R), size=1) * E0[i]
    
 
    s,a0 = pick_inout_hull(S,R,E0,inout=False,di=dira)
    """
    a0 = np.array([[4.99834805e-08, 2.03386701e-07, 2.46629819e-07],
       [1.01684067e-07, 1.67332130e-07, 2.30983803e-07],
       [6.05302887e-08, 4.34798688e-08, 3.95989843e-07],
       [8.25270958e-09, 2.54191656e-07, 2.37555634e-07],
       [4.35351033e-08, 1.93898529e-07, 2.62566368e-07],
       [6.02035068e-09, 1.99224709e-07, 2.94754940e-07],
       [1.03432182e-08, 2.84359600e-07, 2.05297182e-07],
       [2.60515265e-08, 2.18423920e-07, 2.55524554e-07],
       [1.60182962e-07, 1.75465146e-07, 1.64351893e-07],
       [1.71402979e-08, 2.30551241e-07, 2.52308461e-07]])
"""
    n0 = np.random.uniform(10e6, 10e6, S)
    c0 = np.random.uniform(10e-6, 10e-6, R)
    z0 = np.concatenate((n0, c0, a0.flatten(), E0), axis=None)
    
    #s = np.array([0.05605544, 0.02974847, 0.02245477])
    z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))
    
    n = z[:,0:S]
    c = z[:,S:S+R]
    a = z[:,S+R:S+R+S*R]
    a = np.reshape(a,(num_t,S,R))
    #E = z[:,S+R+S*R:2*S+R+S*R]
    
    n_eq = n[-1,:]
    #c_eq = c[-1,:]
    a_eq = a[-1,:,:]
    cent = get_fd_and_centroid(a_eq)[1]
    dist_cent = np.linalg.norm(0.5-np.array(cent))
    #plot_simplex_1frame(a0,s,0)
    num_frames=20
    #plot_gif(a,t,s,num_frames)
    plot_gif(a,n,t,s,num_frames)
    
    
    
