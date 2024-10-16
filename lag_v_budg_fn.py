#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:13:54 2021

@author: brendonmcguinness
"""

# lag vs enzyme budget

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
import random
from scipy.optimize import linprog
from mpl_toolkits import mplot3d
from scipy.spatial import ConvexHull
from scipy import stats
import itertools
from shapely.geometry import Polygon
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.optimize import approx_fprime
from scipy.linalg import lstsq
from scipy.optimize import nnls
from scipy.misc import derivative
from scipy.optimize import lsq_linear
from scipy.spatial.distance import cosine

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm





def monod(c, k):
    return c/(k+c)

def typeIII(c, k, n):
    return (c**n)/(k+(c**n))
    
def constraint(a, Q, dlta, sig):
    return sum(a[sig, :])/(Q[sig]*dlta[sig])-1


def heavyside_approx(a, Q, dlta, sig):
    """
    if constraint(a,E,sig) < 0:
        return 0.0
    else:
        return 1.0


    if constraint( a, E, sig) >0:
        print("contraint is > 0")
        print(constraint( a, E, sig))
        return 1.0
    else:
        return math.exp(10e5 * constraint(a, E, sig)) 

    """
    try:
        ans = 1/(1+math.exp(- (constraint(a, Q, dlta, sig) * 1000000)))
    except OverflowError:
        ans = 1/float('inf')
    return ans


def growth(c, K, a, v, sigma):
    return sum(v * monod(c, K) * a[sigma, :])


def model_noADAPT(y, t, S, R, a, v, dlta, s, u, K):

    n = y[0:S]
    c = y[S:S+R]

    dcdt = np.zeros(R, dtype=float)
    dndt = np.zeros(R, dtype=float)

    for i in range(0, R):
        dcdt[i] = s[i] - (sum(n * a[:, i]) * monod(c[i], K[i])) - u[i]*c[i]
    for sig in range(S):
        dndt[sig] = (growth(c, K, a, v, sig) - dlta[sig])*n[sig]

    dzdt = np.concatenate((dcdt, dndt), axis=None)
    return dzdt


def model(y, t, S, R, v, d, dlta, s, u, K, Q, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    
    
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]
    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                1/R*np.sum(v*monod(c, K)*a[sig, :])))

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))


#    for sig in range(S):
#        for i in range(R):
#            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
#                1/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))


    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def model_sublinear(y, t, S, R, v, d, dlta, s, u, K, Q, k, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter
    k : scaling parameter growth

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = (np.power(n[sig],k)*( np.sum(v * monod(c, K) * a[sig, :]))) - (n[sig] * dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def model_sublinear_noplast(y, t, S, R, v, dlta, s, u, K, Q, k, fr=monod):
    """
    runs model from Pacciani paper no adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter
    k : scaling parameter growth

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]
    #get rid of Nans
    n = np.nan_to_num(n,posinf=0,neginf=0).clip(min=0)
    
    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = (np.power(n[sig],k)*( np.sum(v * monod(c, K) * a[sig, :]))) - (n[sig] * dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def model_when_even(y, t, S, R, v, d, dlta, s, u, K, Q, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    when = y[S+R+S*R:S+R+S*R+1]
    even = y[S+R+S*R+1:S+R+S*R+2]
    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))
    
    dwhendt = full_point_in_hull(s,a)
    devendt = shannon_diversity(n)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(),dwhendt,devendt), axis=None)
    return dzdt
def model_no_E(y, t, S, R, v, d, dlta, s, u, K, Q, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    #E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    #dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))

    #dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten()), axis=None)
    return dzdt

def model_jocob(y):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    
    def monod(c, k):
        return c/(k+c)

    def heavyside_approx(a, Q, dlta, sig):
        """
        if constraint(a,E,sig) < 0:
            return 0.0
        else:
            return 1.0
    
    
        if constraint( a, E, sig) >0:
            print("contraint is > 0")
            print(constraint( a, E, sig))
            return 1.0
        else:
            return math.exp(10e5 * constraint(a, E, sig)) 
    
        """
        try:
            ans = 1/(1+math.exp(- (constraint(a, Q, dlta, sig) * 1000000)))
        except OverflowError:
            ans = 1/float('inf')
        return ans
        
    S = 10
    R = 3
    
    v = np.random.uniform(10e7, 10e7, R)
    dlta = np.random.uniform(5e-3, 5e-3, S) 
    Q = np.ones(S)*1e-5 #10e-5
    #self.Q = np.random.uniform(1e-6,1e-5,S)
    #E0 = np.random.uniform(Q*dlta, Q*dlta, S)
    K = np.random.uniform(1e-4, 1e-4, R) #10e-6
    u = np.zeros(R)
    s = np.random.uniform(10e-5,10e-2,R)
    d = 50e-7*np.ones(S) 
    
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)

    for sig in range(0, S):
        print(c)
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))


    dzdt = np.concatenate((dndt, dcdt, dadt.flatten()), axis=None)
    return dzdt

def model_selfinter(y, t, S, R, v, d, dlta, s, u, K, Q, eps, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter
    eps : self interaction term

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig] - eps[sig]*n[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def model_nonlinear_tradeoffs(y, t, S, R, v, d, dlta, s, u, K, Q, gamma, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]
    #dlta = y[2*S+R+S*R:3*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)
    dddt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp
    
    growth = np.zeros((S,R))
    growthi = np.zeros((S,R))

    for sig in range(0, S):
        growth[sig,:] = (v * monod(c, K) * a[sig, :])
        dndt[sig] = n[sig] * ((np.sum(np.power(growth[sig,:].clip(min=0),gamma))) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))
    #try to go back to when it works only gamma power in last minus term
    for sig in range(S):
        for i in range(R):
            growthi[sig,i] = v[i]*monod(c[i], K[i])*a[sig, i]
            #v[i]*monod(c[i], K[i])*a[sig, i]
            growth_temp = max(growthi[sig,i],0)
            dadt[sig, i] = d[sig]*dlta[sig]*(np.power(growth_temp,gamma) - (
                #heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))
                heavyside_approx(a, Q, dlta, sig)/np.sum(np.power(a[sig, :].clip(min=0),gamma))*np.sum(np.power(growth[sig,:].clip(min=0),gamma))))

    dEdt = np.sum(dadt, 1)
    
    #dlta =  E / Q

    #dddt = E/Q #dEdt
    #print('E shape',E.shape)
    #print('dlta shape',dddt.shape)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def model_type3(y, t, S, R, v, d, dlta, s, u, K, Q, nn, fr=typeIII):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * fr(c, K, nn) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (fr(c[i], K[i], nn) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*fr(c[i], K[i], nn) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*fr(c, K, nn)*a[sig, :])))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt


def model_type1(y, t, S, R, v, d, dlta, s, u, Q, e):
    """
    runs model from Pacciani paper adaptive metabolic strategies but with type 1 functional response


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter
    e : converts amount of excess resource into growth rate of consumer sigma units = 1/resource conc. = mL/g of resource

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * c * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (c[i] * np.sum(n * a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*c[i] - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*c*a[sig, :])))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt


def model_stocha(y, t, S, R, v, d, dlta, s, u, K, Q, fr=monod):
    """
    runs model from Pacciani paper adaptive metabolic strategies but with stochasticity in metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter
    Q : scaling parameter

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :]))) + np.random.normal(loc=0,scale=a.mean()/(S**2))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def model_type1_batch(y, t, S, R, v, d, dlta, s, Q, e, tau):
    """
    runs model from Pacciani paper adaptive metabolic strategies but with type 1 functional response


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    K : Monod parameter
    Q : scaling parameter
    e : converts amount of excess resource into growth rate of consumer sigma units = 1/resource conc. = mL/g of resource
    tau : dilution rate 1/time

    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    for sig in range(0, S):
        # add efficiency term i think after working out units
        dndt[sig] = e[sig]*n[sig] * \
            (np.sum(v * c * a[sig, :]) - dlta[sig]) - tau*n[sig]

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = tau*(s[i]-c[i]) - (c[i] * np.sum(n * a[:, i]))

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*e[sig]*d[sig]*dlta[sig]*(v[i]*c[i] - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*c*a[sig, :])))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt


def model_s_changing(y, t, S, R, v, d, dlta, s1, s2, u, K, Q):
    """
    runs model from Pacciani paper adaptive metabolic strategies


    Parameters
    ----------
    y : array keeping all the state variables
    t : time array
    S : num of species
    R : num of resources
    v : nutritional value of resource i
    d : adaptation rate
    dlta : death rate of species sigma / outflow of chemostat
    s : supply of resource i
    u : degradation rate of resource i
    K : Monod parameter


    Returns
    -------
    dzdt : state variables all packed into same array

    """
    n = y[0:S]
    c = y[S:S+R]
    a = y[S+R:S+R+S*R]
    E = y[S+R+S*R:2*S+R+S*R]

    a = np.reshape(a, (S, R))
    tau = 100
    if ((t//tau) % 2) == 0:
        s = s1
    else:
        s = s2

    dndt = np.zeros(S, dtype=float)
    dcdt = np.zeros(R, dtype=float)
    dadt = np.zeros(shape=(S, R), dtype=float)
    dEdt = np.zeros(S, dtype=float)

    for sig in range(0, S):
        dndt[sig] = n[sig] * (np.sum(v * monod(c, K) * a[sig, :]) - dlta[sig])

    for i in range(0, R):
        # assuming no resource outflow (u is 0 for all i)
        dcdt[i] = s[i] - (monod(c[i], K[i]) * np.sum(n *
                          a[:, i])) - u[i]*c[i]  # -10*c[i]

    # for sig in range(0,S):
    #    dadt[sig,:] = a[sig,:]*d*dlta[sig] * (v*monod(c,K) - (heavyside_approx(a,E,sig)/np.sum(a[sig,:])*np.sum(v*monod(c,K)*a[sig,:])))

    for sig in range(S):
        for i in range(R):
            dadt[sig, i] = a[sig, i]*d[sig]*dlta[sig]*(v[i]*monod(c[i], K[i]) - (
                heavyside_approx(a, Q, dlta, sig)/np.sum(a[sig, :])*np.sum(v*monod(c, K)*a[sig, :])))

    dEdt = np.sum(dadt, 1)

    dzdt = np.concatenate((dndt, dcdt, dadt.flatten(), dEdt), axis=None)
    return dzdt

def compute_jacobian(func, y, *args):
    num_vars = len(y)
    num_outputs = len(func(y, 0, *args))
    jacobian = np.zeros((num_outputs, num_vars))

    # Compute the Jacobian using finite differences
    for i in range(num_vars):
        jacobian[:, i] = approx_fprime(y, lambda y: func(y, 0, *args)[i], epsilon=1e-6) #change to make smaller
        #jacobian[:, i] = derivative(lambda y: func(y, 0, *args)[i],y, dx=1e-6,args=(*args)) #change to make smaller

    return jacobian

def compute_jacobian_centDiff(func, y, *args):
    num_vars = len(y)
    num_outputs = len(func(y, 0, *args))
    jacobian = np.zeros((num_outputs, num_vars))

    # Compute the Jacobian using finite differences
    for i in range(num_vars):
        jacobian[:, i] = approx_fprime(y, lambda y: func(y, 0, *args)[i], epsilon=1e-10) #change to make smaller
        #jacobian[:, i] = derivative(lambda y: func(y, 0, *args)[i],y, dx=1e-6) #change to make smaller

    return jacobian

def shannon_diversity(counts):
    p = counts / np.sum(counts)
    H = - np.sum(p*np.log2(p))

    if math.isnan(H):
        return 0
    else:
        return H
    
def shannon_diversityNan(counts):
    p = counts / np.sum(counts)
    p_pos = p[p>0]
    H = - np.nansum(p_pos*np.log2(p_pos))

    return H

def orderParameter(n):
    numt, S = n.shape
    diff = np.zeros((numt,S))
    #need to decide whether to normalize or not
    for i in range(S):
        diff[:,i] = np.gradient(n[:,i])
    return diff.var(axis=1)
        
def orderParameterCV(n):
    numt, S = n.shape
    diff = np.zeros((numt,S))
    #need to decide whether to normalize or not
    for i in range(S):
        diff[:,i] = np.gradient(n[:,i] / n.sum(axis=1))
    return diff.var(axis=1)
        
def plot2Dsimplex(shat, ahat, string='Equilibrium'):
    S, R = ahat.shape
    plt.figure()

    a_hat_eq_1 = ahat[:, 0]*20

    plt.hlines(1, 1, 20, colors='black')  # Draw a horizontal line
    plt.xlim(0, 21)
    plt.ylim(0.5, 1.5)

    y = np.ones(np.shape(a_hat_eq_1))   # Make all y values the same
    y2 = np.ones(np.shape(shat[0]))

    for sig in range(S):
        # Plot a line at each location specified in a
        plt.plot(a_hat_eq_1[sig]+1, y[sig], '.', ms=30,
                 label='\N{GREEK SMALL LETTER SIGMA}='+str(sig+1))

    plt.plot(shat[0]*20+1, y2, '*', ms=20,
             label='Supply Vector', color='black')
    plt.axis('off')
    plt.legend()
    titl = string + \
        " rescaled \N{GREEK SMALL LETTER ALPHA}'s and supply vector"
    plt.title(titl)
    #plt.xticks(np.arange(0, 20, step=20))
    plt.show()


def isSupplyinConvexHull2D(shat, ahat):
    S, R = ahat.shape
    count1 = 0
    count2 = 0
    for sig in range(S):
        if shat[0] < ahat[sig, 0]:
            count1 += 1
        if shat[0] > ahat[sig, 0]:
            count2 += 1
    if count1 == S or count2 == S:
        return False
    else:
        return True


def acclimation_fitness(s0hat, a0hat, shat, ahat):
    S, R = ahat.shape
    fit_gain = np.zeros(S)
    for sig in range(S):
        fit_gain[sig] = np.sum(np.absolute(s0hat-a0hat[sig, :])) - \
            np.sum(np.absolute(shat-ahat[sig, :]))

    return fit_gain


def initial_dist(s0hat, a0hat):
    S, R = a0hat.shape
    init_dist = np.zeros(S)
    for sig in range(S):
        init_dist[sig] = np.sum(np.absolute(s0hat-a0hat[sig, :]))
    return init_dist


def average_spacing(ahat):
    S, R = ahat.shape
    space_mat = np.zeros((R, S, S))
    for i in range(S):
        for j in range(S):
            space_mat[:, i, j] = np.mean(np.absolute(ahat[i, :]-ahat[j, :]))
    return space_mat.mean(axis=None)


def get_hats(n_eq, a_eq, a0, dlta, v, s, E0, Q):

    S, R = a_eq.shape

    if isinstance(Q, np.ndarray):
        # chill
        abcd = 1
    else:
        temp = Q*np.ones(S)
        Q = temp

    # build convex hull equation
    x = n_eq * dlta / np.sum(n_eq*dlta)

    # for s_hat now using other definition
    s0_hat = v * s / np.sum(v*s)

    a_hat_eq = np.zeros((S, R))
    a0_hat = np.zeros((S, R))

    s_hat = np.zeros(R)

    for i in range(R):

        for sig in range(S):

            a_hat_eq[sig, i] = a_eq[sig, i] / (Q[sig]*dlta[sig])
        a0_hat[:, i] = a0[:, i] / (E0)
        s_hat[i] = np.sum(x * a_hat_eq[:, i])

    return s0_hat, a0_hat, s_hat, a_hat_eq


def get_avg_accfit_per_rank(ranks, acc_fit):

    N = ranks.shape[0]
    S = ranks.shape[1]

    avg_accfit = np.zeros(S)
    std_accfit = np.zeros(S)
    for k in range(S):
        temp_arr = []
        for j in range(N):
            temp_arr.append(acc_fit[j, int(ranks[j, k])])
        # took out abs on mean and std
        # print(temp_arr)
        avg_accfit[k] = np.mean(temp_arr)
        std_accfit[k] = np.std(temp_arr)
    return avg_accfit, std_accfit
#    for sig in range(S):
 #       idx =


def get_avg_initdist_per_rank(ranks, init_dist):

    N = ranks.shape[0]
    S = ranks.shape[1]

    avg_initdist = np.zeros(S)
    std_initdist = np.zeros(S)
    for k in range(S):
        temp_arr = []
        for j in range(N):
            temp_arr.append(init_dist[j, int(ranks[j, k])])
        avg_initdist[k] = np.mean(temp_arr)
        std_initdist[k] = np.mean(temp_arr)
    return avg_initdist, std_initdist


def get_rank_dist(n_eq):
    ranks = np.argsort(n_eq)[::-1]
    rd = n_eq[ranks]
    rd[rd < 0] = 1e-30
    rdlog = np.log10(rd)
    #rdlog[np.isnan(rdlog)] = -30
    return rdlog


def get_rank_dist_nolog(n_eq):
    ranks = np.argsort(n_eq)[::-1]
    rd = n_eq[ranks]
    rd[rd < 0] = 1e-30
    #rdlog[np.isnan(rdlog)] = -30
    return rd


def get_rank_dist_save_ind(n_eq):
    ranks = np.argsort(n_eq)[::-1]
    rd = n_eq[ranks]
    rd[rd < 0] = 1e-60
    #rdlog = np.log10(rd)
    idx = np.arange(n_eq.size)
    ind = idx[np.argsort(n_eq[idx])[::-1]]
    return rd, ind


def avg_off_diagonal(p):
    # assumes p is square matrix with diag elements set to 0
    S = p.shape[0]
    return np.sum(p)/(S**2-S)


def calc_niche_overlap(a):
    # bias to strategy close to either middle (bad) or supply vector(good)
    S = a.shape[0]
    pij = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            pij[i, j] = np.sum(a[i, :]*a[j, :]) / \
                np.sqrt(np.sum(a[i, :]**2)*np.sum(a[j, :]**2))
    diag = pij.diagonal()
    np.fill_diagonal(pij, 1-diag)
    #avg_p = np.average(np.average(pij))
    return avg_off_diagonal(pij)

#a0 = np.array([[0,1],[1,0]])
# print(calc_niche_overlap(a0))


def opt_fun_d(tau, v, s, c_eq):
    return tau*np.sum(v*(s*np.log(s/c_eq)-(s-c_eq)))


def calc_CVs(n_eqs, c_eqs, a_eqs):

    CV_n = n_eqs.std(axis=0)/n_eqs.mean(axis=0)
    CV_c = c_eqs.std(axis=0)/c_eqs.mean(axis=0)
    CV_a = a_eqs.std(axis=0).mean()/a_eqs.mean()

    return CV_n, CV_c, CV_a


def plot_max_diff_commun(n, n_eqs, c, c_eqs, a, Q, dlta, s, v, t, N, title='Self organized population densities'):

    clrs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    S = a.shape[2]
    R = a.shape[3]

    dist = np.zeros((N, N))
    dist_c = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            dist[i, j] = np.linalg.norm(n_eqs[i, :] - n_eqs[j, :])
            dist_c[i, j] = np.linalg.norm(c_eqs[i, :] - c_eqs[j, :])

    index = np.where(dist == np.amax(dist))
    maxi = index[0][0]
    maxj = index[0][1]

    cindex = np.where(dist_c == np.amax(dist_c))
    cmaxi = cindex[0][0]
    cmaxj = cindex[0][1]

    plt.figure()
    for sig in range(S):
        plt.plot(t, n[maxi, :, sig], color=clrs[sig % len(clrs)])
        plt.plot(t, n[maxj, :, sig], linestyle='dashed',
                 color=clrs[sig % len(clrs)])
    plt.title(title)
    plt.ylabel('Consumer Density')
    plt.xlabel('time')
    plt.show()

    """
    plt.figure()
    for i in range(R):
        plt.plot(t,c[cmaxi,:,i],color=clrs[i%len(clrs)])
        plt.plot(t,c[cmaxj,:,i],linestyle='dashed',color=clrs[i%len(clrs)])
    plt.title('Resource concentration over time in two communities with different IC')
    plt.ylabel('Resource Conc.')
    plt.xlabel('time')"""

    plt.figure()
    ls = ['solid', 'dashed', 'dotted']

    ahat_t1 = np.zeros((t.shape[0], S, R))
    ahat_t2 = np.zeros((t.shape[0], S, R))
    rescaled_sup = v*s/np.sum(v*s)
    for sig in range(S):
        for i in range(R):
            lab = '\N{GREEK SMALL LETTER SIGMA}='+str(sig+1)+', i='+str(i+1)
            ahat_t1[:, sig, i] = a[maxi, :, sig, i] / (Q[sig]*dlta[sig])
            ahat_t2[:, sig, i] = a[maxj, :, sig, i] / (Q[sig]*dlta[sig])
            plt.plot(t, ahat_t1[:, sig, i], label=lab,
                     color=clrs[sig % len(clrs)], linestyle=ls[i % len(ls)], linewidth=1)
            plt.plot(t, ahat_t2[:, sig, i], label=lab,
                     color=clrs[sig % len(clrs)], linestyle=ls[i % len(ls)], linewidth=0.5)
    # plt.legend()
    for i in range(R):
        plt.plot(t, rescaled_sup[i]*np.ones(t.shape[0]),
                 color='k', linestyle=ls[i % len(ls)], linewidth=2)
    plt.title(
        'Alphas for Strategy \N{GREEK SMALL LETTER SIGMA} and Resource i vs time')
    plt.ylabel(
        '$\N{GREEK SMALL LETTER ALPHA}_{\N{GREEK SMALL LETTER SIGMA}i}(t)$')
    plt.xlabel('$t$')
    plt.ylim(0, 1)
    plt.show()


def get_cartesian_from_barycentric(b, t):
    return t.dot(b)


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def full_point_in_hull(s, a0):
    R = s.shape[0]
    # cant do convex rhull because we can represent simplex in N-1 dimensions
    #trying basis vector on stack exchange
    if R>2:
        return point_in_hull(bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0], ConvexHull(bary2cart(a0,corners=simplex_vertices(R-1))[0]))
    else:
        #R==2
        return not ((s[0] > a0[:,0]).all() or (s[0] < a0[:,0]).all())

def polycorners(ncorners=3):
    '''
    Return 2D cartesian coordinates of a regular convex polygon of a specified
    number of corners.
    Args:
        ncorners (int, optional) number of corners for the polygon (default 3).
    Returns:
        (ncorners, 2) np.ndarray of cartesian coordinates of the polygon.
    '''

    center = np.array([0.5, 0.5])
    points = []

    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))

    return np.array(points)

def tetrahedron_corners(ncorners=4):
    vertices = np.array([[np.sqrt(8/9),0,-1/3],[-np.sqrt(2/9),np.sqrt(2/3),-1/3],[-np.sqrt(2/9),-np.sqrt(2/3),-1/3],[0,0,1]])
    return vertices

    
def v(n):
    '''returns height of vertex for simplex of n dimensions
    '''
    return np.sqrt((n+1)/(2*n))

      
def simplex_vertices(n):
    '''
    from https://www.tandfonline.com/doi/pdf/10.1080/00207390110121561?needAccess=true
    maybe or maybe not scaled by sqrt(3)/2
    Parameters
    ----------
    n : number of vertices

    Returns
    -------
    vert : vertices of simplex in R^n+1 cartesian space

    
    '''
    vert = np.zeros((n+1,n))
    for i in range(n+1):
        for j in range(n):
            if i - j == 1:
                vert[i,j] = np.sqrt(3)/2 * np.sqrt((j+2)/(2*(j+1))) #v(j+1)
            if i - j > 1:
                vert[i,j] = np.sqrt(3)/2 * np.sqrt((j+2)/(2*(j+1))) / (j+2) # v(j+1)/(j+2)
    return vert
 
       


def bary2cart(bary, corners=None):
    '''
    Convert barycentric coordinates to cartesian coordinates given the
    cartesian coordinates of the corners.
    Args:
        bary (np.ndarray): barycentric coordinates to convert. If this matrix
            has multiple rows, each row is interpreted as an individual
            coordinate to convert.
        corners (np.ndarray): cartesian coordinates of the corners.
    Returns:
        2-column np.ndarray of cartesian coordinates for each barycentric
        coordinate provided.
    '''

    if np.array((corners == None)).any():
        corners = polycorners(bary.shape[-1])

    cart = None

    if len(bary.shape) > 1 and bary.shape[1] > 1:
        cart = np.array([np.sum(b / np.sum(b) * corners.T, axis=1)
                        for b in bary])
        
    else:
        cart = np.sum(bary / np.sum(bary) * corners.T, axis=1)

    return cart,corners

def cart2bary(cart):
    #can only do triangle
    x = cart[0]
    y = cart[1]
    x1,x2,x3 = simplex_vertices(2).T[0]
    y1,y2,y3 = simplex_vertices(2).T[1]
        
    l1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
    l2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3)) / ((y2-y3)*(x1-x3) + (x3-x2)*(y1-y3))
    l3 = 1-l1-l2
    return l1,l2,l3

def ndim_simplex(n):
    vert = np.zeros((n+1,n+1)) 
    e = np.identity(n+1)
    for i in range(n+1):
        vert[i,:] = 1/(np.sqrt(2))*e[i,:] + (1/(n*np.sqrt(2))*(1+1/np.sqrt(n+1))*np.ones(n+1)) + 1/np.sqrt(2*(n+1))*np.ones(n+1)
    return vert



def slope(n1, n2, t1, t2):
    return (n2-n1) / (t2-t1)


def find_eqt_of_n(n, t, eps):
    sl = np.zeros(t.shape[0])
    for i in range(t.shape[0]-1):
        sl[i] = slope(n[i], n[i+1], t[i], t[i+1])
    for i in range(t.shape[0]-30):
        if np.sum(np.abs(sl[i:i+30])) < 30*eps*n[i] or np.sum(n[i:i+30]) < 30*eps:
            return i
    return t.shape[0]-1


def community_resilience(n, t, eps):
    S = n.shape[1]
    res = []
    if len(n.shape) == 3:
        S = S*n.shape[2]
        n = n.reshape(*n.shape[:-2], -1)
        res = []        
        #print(S)
    for sig in range(S):
        if n[-1, sig] > 1.0:
            res.append(t[find_eqt_of_n(n[:, sig], t, eps)])
    return np.mean(res)

def avg_eq_time(c,t,rel_tol=0.001):
    stable_index = np.zeros(c.shape[1])
    for i in range(c.shape[1]):
        try:
            #try
            stable_index[i] = np.where(np.abs(c[-1,i] - c[:,i]) > (c[-1,i]*rel_tol))[0][-1] + 1                
        except:
            print('not finding eq time')
            stable_index[i] = t[-1] #maybe delete this
    return np.mean(t[stable_index.astype(int)])

def avg_eq_time_traits(a,t,rel_tol=0.001):
    S,R = a.shape[1:]
    stable_index = np.zeros((S,R))
    for i in range(S):
        for j in range(R):
            stable_index[i,j] = np.where(np.abs(a[-1,i,j] - a[:,i,j]) > (a[-1,i,j]*rel_tol))[0][-1] + 1
    return np.mean(t[stable_index.flatten().astype(int)])
        
def log_series_pmf(k,p):
    """
    returns log series from 1...k

    Parameters
    ----------
    k : array from 1...N_S
    p : parameter between 0 and 1
   

    Returns
    -------
    pmf : log series pmf

    """
        
    pmf = np.zeros(k.shape[0])
    for i in range(k.shape[0]):
        pmf[i] = -1 / np.log(1-p+1e-9) * (p ** (k[i])) / k[i]
    return pmf

def log_normal_pmf(x,u,sig):
    """
    returns log series from 1...k

    Parameters
    ----------
    x : array from 1...N_S
    u : mean
    sig : variance
   

    Returns
    -------
    pmf : log normal pmf

    """
        
    pmf = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        pmf[i] = 1 / (x[i]*sig*math.sqrt(2*math.pi)) * math.exp(-(np.log(x[i]-(u**2)))/(2*(sig**2)))
    return pmf
        
def pick_inout_hull(S, R, E0, a=10e-6, b=10e-2, inout=False,di=None):
    if di.any() == None:
        di = np.ones(R)
    a0 = np.zeros((S, R), dtype=float)
    for i in range(0, S):
        a0[i, :] = np.random.dirichlet(di*np.ones(R), size=1) * E0[i]
    a0_scaled = a0/E0[:, None]
    s = np.random.uniform(a, b, R)
    if full_point_in_hull(s, a0_scaled) == inout:
        return s, a0
    else:
        return pick_inout_hull(S, R, E0, inout=inout,di=di)
    
def pick_inout_hull_a0(s, S, R, E0, inout=True):
    a0 = np.zeros((S, R), dtype=float)
    for i in range(0, S):
        #dira = np.random.randint(1,dr+1,size=R)
        a0[i, :] = np.random.dirichlet(np.ones(R), size=1) * E0[i]
    a0_scaled = a0/E0[:, None]
    if full_point_in_hull(s, a0_scaled) == inout:
        return a0
    else:
        return pick_inout_hull_a0(s, S, R, E0, inout=inout)


def pick_inout_hull_s(E0, a0, a=10e-6, b=10e-2, inout=True):
    S, R = a0.shape
    a0_scaled = a0/E0[:, None]
    s = np.random.uniform(a, b, R)
    if full_point_in_hull(s, a0_scaled) == inout:
        return s
    else:
        return pick_inout_hull_s(E0, a0, inout=inout)
    

def missclass(mode, ind):
    mc = np.zeros(ind.shape[0])
    for j in range(ind.shape[0]):
        for i in range(ind.shape[1]):
            if ind[i, j] != mode[j]:
                mc[j] += 1
    return mc


def get_rank_key(mode, key):
    ix, = np.where(mode.flatten() == np.argmax(key))
    if ix.size > 0:
        return ix[0]
    return get_rank_key(mode, np.delete(key, np.argmax(key)))


def keystones(S, R, v, d, dlta, s, u, K, Q, t):
    num_t = t.shape[0]
    E0 = Q*dlta
    # as of now the initial stragies are random in each community/remember to change back
    a0 = np.zeros((S, R), dtype=float)
    for i in range(0, S):
        a0[i, :] = np.random.dirichlet(np.ones(R), size=1) * E0[i]
    # draw fresh supplies
    s = np.random.uniform(10e-6, 10e-2, R)
    s, a0 = pick_inout_hull(S, R, E0, inout=False)  # toggle
    n0 = np.random.uniform(10e5, 10e6, S)
    c0 = np.random.uniform(10e-3, 10e-2, R)

    z0 = np.concatenate((n0, c0, a0.flatten(), E0), axis=None)
    z = odeint(model, z0, t, args=(S, R, v, d, dlta, s, u, K, Q))

    # for with d non zero
    n = z[:, 0:S]
    c = z[:, S:S+R]
    a_temp = z[:, S+R:S+R+S*R]
    a = np.reshape(a_temp, (num_t, S, R))

    n_eqs = n[-1, :]
    c_eqs = c[-1, :]
    a_eqs = a[-1, :, :]

    nres = np.zeros((S, S))
    ares = np.zeros((S, S, R))
    distn = np.zeros(S)
    dista = np.zeros(S)
    tau = np.zeros(S)

    rank_abund = np.zeros((S, S))
    ind = np.zeros((S, S))

    for j in range(S):

        n_p = np.ones(S)
        n_p[j] = 0
        zp = np.concatenate((n_p*n_eqs, c_eqs, a_eqs.flatten(), E0), axis=None)

        z = odeint(model, zp, t, args=(S, R, v, d, dlta, s, u, K, Q))

        nt = z[:, 0:S]
        ct = z[:, S:S+R]
        at_temp = z[:, S+R:S+R+S*R]
        at = np.reshape(at_temp, (num_t, S, R))

        nres[j, :] = nt[-1, :]
        ares[j, :, :] = at[-1, :, :]

        # per capita
        distn[j] = np.linalg.norm(nres[j, :]-n_eqs)  # /n_eqs[j]
        dista[j] = np.linalg.norm(ares[j, :]-a_eqs)  # /n_eqs[j]

        tau[j] = community_resilience(nt, t, 10e-6)  # /n_eqs[j]

        # get ranks
        rank_abund[j, :], ind[j, :] = get_rank_dist_save_ind(nres[j, :])

    plt.figure()
    plt.plot(t, nt)

    trait_var = np.zeros(S)
    for sig in range(S):
        trait_var[sig] = (ares[:, sig, :]/(Q[0]*dlta[0])).var(axis=1).mean()

    #mode,count = stats.mode(ind[:,0:S-2])
    mode, count = stats.mode(ind)

    indforhist = get_rank_key(mode, distn)
    indforhista = get_rank_key(mode, dista)
    indforhistt = get_rank_key(mode, tau)
    indforhisttv = get_rank_key(mode, trait_var)

    mode = mode.flatten()

    return np.argmax(distn), np.argmax(dista), np.argmax(tau), np.argmax(trait_var), indforhist, indforhista, indforhistt, indforhisttv, ind, mode


def keystones_cat(S, R, v, d, dlta, s, u, K, Q, t, z0):
    

    while True:
        try:
            num_t = t.shape[0]
            E0 = Q*dlta
        
            # run model
            z = odeint(model, z0, t, args=(S, R, v, d, dlta, s, u, K, Q))
        
            # for with d non zero
            n = z[:, 0:S]
            c = z[:, S:S+R]
            a_temp = z[:, S+R:S+R+S*R]
            a = np.reshape(a_temp, (num_t, S, R))
        
            n_eqs = n[-1, :]
            c_eqs = c[-1, :]
            a_eqs = a[-1, :, :]
            init_shift_trait = np.linalg.norm((a_eqs/E0[:, None])-(a[0,:,:]/E0[:, None]))
            # could be worth getting rank before actually
            r, ranked_ind = get_rank_dist_save_ind(n_eqs)
            
            
                          
            fd_og, cent_og, insur_og = get_fd_and_centroid(a_eqs)
            cen,dn = get_cen_dn(a_eqs,E0,s)
           
        
            nres = np.zeros((S, S))
            ares = np.zeros((S, S, R))
            distn = np.zeros(S)
            dista = np.zeros(S)
            tau = np.zeros(S)
            dns = np.zeros(S)
            das = np.zeros(S)
            ts = np.zeros(S)
            rank_abund = np.zeros((S, S))
            ind = np.zeros((S, S))
            fd = np.zeros(S)
            cent = np.zeros((S,2))
            insur = np.zeros(S)
            diff_RAD = np.zeros(S)
            area_overlap = np.zeros(S)

        
            for j in range(S):
        
                n_p = np.ones(S)
                n_p[j] = 0
                zp = np.concatenate((n_p*n_eqs, c_eqs, a_eqs.flatten(), E0), axis=None)
        
                z = odeint(model, zp, t, args=(S, R, v, d, dlta, s, u, K, Q))
        
                nt = z[:, 0:S]
                ct = z[:, S:S+R]
                at_temp = z[:, S+R:S+R+S*R]
                at = np.reshape(at_temp, (num_t, S, R))
        
                nres[j, :] = nt[-1, :]
                ares[j, :, :] = at[-1, :, :]
        
                # per capita
                # need to accomadate this for no acclimation because we're getting over/underflow
                # just do for species that have >x biomass at equilibrium
                ind_nonzeros = np.argwhere(n_eqs > 10e-4).flatten()
                if j in ind_nonzeros:
                    distn[j] = np.linalg.norm(nres[j, :]-n_eqs) #/ n_eqs[j]
                    #scaled per capita
                    dns[j] = distn[j] / n_eqs[j]
        
                dista[j] = np.linalg.norm((ares[j, :]/E0[:, None])-(a_eqs/E0[:, None])) #/ np.abs(n_eqs[j])
                das[j] = dista[j] / np.abs(n_eqs[j])
                tau[j] = community_resilience(nt, t, 10e-6) #/ n_eqs[j]
                ts[j] = tau[j] / n_eqs[j]
        
                # get ranks
                rank_abund[j, :], ind[j, :] = get_rank_dist_save_ind(nres[j, :])
                #rk_ind = get_rank_scores(ind)
                mode, count = stats.mode(ind)
                mode = mode.flatten()
                
                #community metric of howdifferent RADs are
                diff_RAD[j] = np.linalg.norm(rank_abund[j,:] - r)
                area_overlap[j] = get_area_intersect(ares[j,:,:], a_eqs)

                #get functional diversity and centroid
                #print(ares[j,:,:])
                #print(np.delete(ares[j,:,:],j,axis=0))
                fd[j],cent[j,:],insur[j] = get_fd_and_centroid(np.delete(ares[j,:,:],j,axis=0))
                
            trait_var = np.zeros(S)
            for sig in range(S):
                trait_var[sig] = (ares[:, sig, :]/(Q[:, None] *
                                  dlta[:, None])).var(axis=1).mean()
            #print(trait_var)
            rtv, tv_ind = get_rank_dist_save_ind(trait_var)
            #print(rtv,tv_ind)
            
            
            
            return distn, dns, dista, das, tau, ts, tv_ind, ranked_ind , trait_var, init_shift_trait, ares, nres, a_eqs, area_overlap, n_eqs, fd, fd_og, cent, cent_og, insur, insur_og, diff_RAD,cen,dn
            break
        except RuntimeWarning:
            print('Caught warning, retrying...')
            continue

def keystones_cat_timeseries(S, R, v, d, dlta, s, u, K, Q, t, z0):
    

    while True:
        try:
            num_t = t.shape[0]
            E0 = Q*dlta
        
            # run model
            z = odeint(model, z0, t, args=(S, R, v, d, dlta, s, u, K, Q))
        
            # for with d non zero
            n = z[:, 0:S]
            c = z[:, S:S+R]
            a_temp = z[:, S+R:S+R+S*R]
            a = np.reshape(a_temp, (num_t, S, R))
        
            n_eqs = n[-1, :]
            c_eqs = c[-1, :]
            a_eqs = a[-1, :, :]
            init_shift_trait = np.linalg.norm((a_eqs/E0[:, None])-(a[0,:,:]/E0[:, None]))
            # could be worth getting rank before actually
            r, ranked_ind = get_rank_dist_save_ind(n_eqs)
            
            fd_og, cent_og, insur_og = get_fd_and_centroid(a_eqs)
        
        
            nres = np.zeros((S, S))
            ares = np.zeros((S, S, R))
            distn = np.zeros(S)
            dista = np.zeros(S)
            tau = np.zeros(S)
            dns = np.zeros(S)
            das = np.zeros(S)
            ts = np.zeros(S)
            rank_abund = np.zeros((S, S))
            ind = np.zeros((S, S))
            fd = np.zeros(S)
            cent = np.zeros((S,2))
            insur = np.zeros(S)
            diff_RAD = np.zeros(S)

            
            area_overlap = np.zeros(S)
        
            for j in range(S):
        
                n_p = np.ones(S)
                n_p[j] = 0
                zp = np.concatenate((n_p*n_eqs, c_eqs, a_eqs.flatten(), E0), axis=None)
        
                z = odeint(model, zp, t, args=(S, R, v, d, dlta, s, u, K, Q))
        
                nt = z[:, 0:S]
                ct = z[:, S:S+R]
                at_temp = z[:, S+R:S+R+S*R]
                at = np.reshape(at_temp, (num_t, S, R))
        
                nres[j, :] = nt[-1, :]
                ares[j, :, :] = at[-1, :, :]
        
                # per capita
                # need to accomadate this for no acclimation because we're getting over/underflow
                # just do for species that have >x biomass at equilibrium
                ind_nonzeros = np.argwhere(n_eqs > 10e-4).flatten()
                if j in ind_nonzeros:
                    distn[j] = np.linalg.norm(nres[j, :]-n_eqs) #/ n_eqs[j]
                    #scaled per capita
                    dns[j] = distn[j] / n_eqs[j]
        
                dista[j] = np.linalg.norm((ares[j, :]/E0[:, None])-(a_eqs/E0[:, None])) #/ np.abs(n_eqs[j])
                das[j] = dista[j] / np.abs(n_eqs[j])
                tau[j] = community_resilience(nt, t, 10e-6) #/ n_eqs[j]
                ts[j] = tau[j] / n_eqs[j]
        
                # get ranks
                rank_abund[j, :], ind[j, :] = get_rank_dist_save_ind(nres[j, :])
                #rk_ind = get_rank_scores(ind)
                mode, count = stats.mode(ind)
                mode = mode.flatten()
                
                #community metric of howdifferent RADs are
                diff_RAD[j] = np.linalg.norm(rank_abund[j,:] - r)
                
                #get functional diversity and centroid
                #print(ares[j,:,:])
                #print(np.delete(ares[j,:,:],j,axis=0))
                fd[j],cent[j,:],insur[j] = get_fd_and_centroid(np.delete(ares[j,:,:],j,axis=0))
                area_overlap[j] = get_area_intersect(ares[j,:,:], a_eqs)
            trait_var = np.zeros(S)
            for sig in range(S):
                trait_var[sig] = (ares[:, sig, :]/(Q[:, None] *
                                  dlta[:, None])).var(axis=1).mean()
            #print(trait_var)
            rtv, tv_ind = get_rank_dist_save_ind(trait_var)
            #print(rtv,tv_ind)
            
            
            
            return distn, dns, dista, das, tau, ts, tv_ind, ranked_ind , trait_var, init_shift_trait, ares, nres, a_eqs, area_overlap, n_eqs, fd, fd_og, cent, cent_og, insur, insur_og, diff_RAD,n #cen,dn # third from back deleted n
            break
        except RuntimeWarning:
            print('Caught warning, retrying...')
            continue

def keystones_cat_meta(S, R, v, d, dlta, s, u, K, Q, t, z0):
    #Does only initial shift trait distance for inout

    while True:
        try:
            num_t = t.shape[0]
            E0 = Q*dlta
        
            # run model
            z = odeint(model, z0, t, args=(S, R, v, d, dlta, s, u, K, Q))
        
            # for with d non zero
            n = z[:, 0:S]
            c = z[:, S:S+R]
            a_temp = z[:, S+R:S+R+S*R]
            a = np.reshape(a_temp, (num_t, S, R))
        
            n_eqs = n[-1, :]
            c_eqs = c[-1, :]
            a_eqs = a[-1, :, :]
            init_shift_trait = np.linalg.norm((a_eqs/E0[:, None])-(a[0,:,:]/E0[:, None]))
            
            return init_shift_trait
            break
        except RuntimeWarning:
            print('Caught warning, retrying...')
            continue
'''old centeroid function
def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length
'''
def centeroidnp(arr):
    length,dim = arr.shape
    summ = np.zeros(dim)
    for i in range(dim):
        summ[i] = np.sum(arr[:, i])
        
    return summ/length

def weighted_centroid(a,w):
    length,dim = a.shape
    summ = np.zeros(dim)
    for i in range(dim):
        summ[i] = np.sum((a[:, i]*w) /w.sum())
        
    return summ #/length
    #R = a.shape[1]
    #ac,corners = bary2cart(a,corners=simplex_vertices(R-1))    
    #return centeroidnp((a*w[:,None])/w.sum())[:,None] 
    
    

def get_fd_and_centroid(a):
        #make simplex
    #ind_del = 4
    #ind_samp = 0
    #get rank of trait point we deleted
    #rank_del = int(df.iloc[ind_samp*10+ind_del]['Rank'])
    #ares_del = np.delete(ares,ind_del,axis=2)
    #to get trait points organized by rank
    #rs,inds = get_rank_dist_save_ind(nres[ind_samp,ind_del,:])
    
    S,R = a.shape
    ac,corners = bary2cart(a,corners=simplex_vertices(R-1))
    ac1,corners1 = bary2cart(a,corners=None)

    if R>S or R<3:
        return -1, centeroidnp(ac), -1
    else:
        hull = ConvexHull(ac)
    
        
        return hull.volume, centeroidnp(ac), S-len(hull.vertices)

def supply_to_centroid(s,a,E0):
    S,R = a.shape
    if a.sum() < S-1:
        a = a/E0[:,None]
    ac = bary2cart(a,corners=simplex_vertices((R-1)))[0]
    sc = bary2cart(s,corners=simplex_vertices((R-1)))[0]
    cent = centeroidnp(ac)
    return np.sqrt(np.sum((cent-sc)**2))

def supply_to_weighted_centroid(s,a,n,E0):
    S,R = a.shape
    if a.sum() < S-1:
        a = a/E0[:,None]
    ac = bary2cart(a,corners=simplex_vertices((R-1)))[0]
    sc = bary2cart(s,corners=simplex_vertices((R-1)))[0]
    cent = weighted_centroid(ac,n)
    return np.sqrt(np.sum((cent-sc)**2))    

def distance(s,a,E0):
    R = s.shape[0]
    sc = bary2cart(s,corners=simplex_vertices((R-1)))[0]
    ac = bary2cart(a/E0[0],corners=simplex_vertices((R-1)))[0]
    return  np.sqrt(np.sum((ac-sc)**2))

def distanceN0(n01,n02):
    return np.sqrt(np.sum((n01-n02)**2))
    
def distanceA0(a01,a02,E0):
    S, R = a02.shape
    ac1 = bary2cart(a01/E0[0],corners=simplex_vertices((R-1)))[0]
    ac2 = bary2cart(a02/E0[0],corners=simplex_vertices((R-1)))[0]
    return np.sqrt(np.sum((ac1-ac2)**2))

def get_area(ac):
    hull = ConvexHull(ac)
    return hull.volume

def get_surface_area(ac):
    hull = ConvexHull(ac)
    return hull.area

def plot_contour(f, x1bound, x2bound, resolution, ax):
    x1range = np.linspace(x1bound[0], x1bound[1], resolution)
    x2range = np.linspace(x2bound[0], x2bound[1], resolution)
    xg, yg = np.meshgrid(x1range, x2range)
    zg = np.zeros_like(xg)
    for i,j in itertools.product(range(resolution), range(resolution)):
        zg[i,j] = f([xg[i,j], yg[i,j]])
    ax.contour(xg, yg, zg, 100)
    return ax


#for finding area of intersection of two convex hulls
def clip(subjectPolygon, clipPolygon):
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

   outputList = subjectPolygon
   cp1 = clipPolygon[-1]

   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]

      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
   return(outputList)

def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    #https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python 
    # from Boris Gorelik
    return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(list_of_xy_coords, centre_of_rotation_xy_coord, clockwise=True):
    cx,cy=centre_of_rotation_xy_coord
    angles = [math.atan2(x-cx, y-cy) for x,y in list_of_xy_coords]
    indices = argsort(angles)
    if clockwise:
        return [list_of_xy_coords[i] for i in indices]
    
def get_area_intersect(a1,a2):
    S,R = a1.shape
    hull1 = ConvexHull(bary2cart(a1,corners=ndim_simplex(R-1))[0])
    hull2 = ConvexHull(bary2cart(a2,corners=ndim_simplex(R-1))[0])
    p1 = list(map(tuple,hull1.points[hull1.vertices]))
    p2 = list(map(tuple,hull2.points[hull2.vertices]))
    #print('hull1 area:',hull1.volume)
    #print('hull2 area:',hull2.volume)
    print(p1)
    poly1=Polygon(p1)
    poly2=Polygon(p2)
    #rotational_sort(p1, (0,0),True)
    points_inter = clip(p1,p2)
    #print(points_inter)
    poly_inter = Polygon(points_inter)
    return poly_inter.area

def get_cen_dn(a_eq,E0,s):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1)) #.sum(axis=1)#a_bar = bary2cart()
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
            
    comp_eff = comp.sum(axis=2).sum(axis=0)
    cen = comp_eff #/comp_eff.max()
    dn = dist #/dist.max()
    #un normalized
    return cen,dn 

def get_comp_std(a_eq,E0,s):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
            
    A = comp.sum(axis=2)
    
    comp_var = np.std(A[A!=0])
    #un normalized
    return comp_var

def comp_dist(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    return comp.sum(axis=2)

def pred_rad_from_traits(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),dist]).T
    Y=n_eq.T
    reg = LinearRegression().fit(X,Y)
    response = reg.predict(X)
    scaled_regcoef = reg.coef_/(np.abs(reg.coef_).max())
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y),(reg.coef_[0]*comp_eff/(S-1))+(dist*reg.coef_[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef

def pred_rad_from_traits_noscale(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    Y=n_eq.T
    reg = LinearRegression(fit_intercept=True,positive=True).fit(X,Y)
    #reg = Ridge(alpha=1.0).fit(X,Y)
    response = reg.predict(X)
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y),(reg.coef_[0]*comp_eff/(S-1))+(dist*reg.coef_[1]),(comp_eff/(S-1)),dist,reg.coef_,reg,response

def pred_rad_from_comp_noscale(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1)]).T
    Y=n_eq.T
    reg = LinearRegression(fit_intercept=True).fit(X,Y)
    #reg = Ridge(alpha=1.0).fit(X,Y)
    response = reg.predict(X)
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y)

def pred_rad_from_dist_noscale(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))

    X = np.array([-dist]).T
    Y=n_eq.T
    reg = LinearRegression(fit_intercept=True).fit(X,Y)
    #reg = Ridge(alpha=1.0).fit(X,Y)
    response = reg.predict(X)
    
#    return reg.score(X,Y),(scaled_regcoef[0]*comp_eff/(S-1))+(dist*scaled_regcoef[1]),(comp_eff/(S-1)).mean(),dist.mean(),scaled_regcoef
    return reg.score(X,Y)

def pred_rad_multiple(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    
    df = pd.DataFrame(X,columns = ['x1','x2'])
    df['y'] = n_eq.T 
    lm = ols('y ~ x1 + x2',df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    scorec = anova_table.loc['x1','sum_sq'] / anova_table['sum_sq'].sum()
    scored = anova_table.loc['x2','sum_sq'] / anova_table['sum_sq'].sum()
    
    return score, scorec, scored

#redo but with cosine distances so its better in high dimensions
def pred_rad_multiple_cosine(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    dist = np.zeros(S)
    
    comp = np.zeros((S,S))
    for i in range(S):
        dist[i] = cosine(a_eq[i,:],s)
        for j in range(S):
            comp[i,j] = cosine(a_eq[i,:],a_eq[j,:])
    comp_eff = comp.sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    
    df = pd.DataFrame(X,columns = ['x1','x2'])
    df['y'] = n_eq.T 
    lm = ols('y ~ x1 + x2',df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    scorec = anova_table.loc['x1','sum_sq'] / anova_table['sum_sq'].sum()
    scored = anova_table.loc['x2','sum_sq'] / anova_table['sum_sq'].sum()
    
    return score, scorec, scored

def pred_ranks(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]

    ranks, ind = get_rank_dist_save_ind(n_eq)
    
    
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),-dist]).T
    
    df = pd.DataFrame(X,columns = ['x1','x2'])
    df['y'] = n_eq.T 
    lm = ols('y ~ x1 + x2',df).fit()
    c,b,a = lm.params
    yy = a*comp_eff - b*dist + c
    indp = get_rank_dist_save_ind(yy)[1]

    return ind, indp, n_eq, yy

def chooseAbundWeights(a,s):
    a = bary2cart(a,corners=simplex_vertices(a.shape[1]-1))[0]
    s = bary2cart(s,corners=simplex_vertices(s.shape[0]-1))[0]
    x = lsq_linear(a.T, s, bounds=(0, 1), lsmr_tol='auto', verbose=0).x
    return x / x.sum()
    

def pickPointInitDist(s, dist, count=0):
    #will only work for R=3
    if count > 500:
        return -1

    rad = np.random.uniform(0, 2*np.pi)
    cart = bary2cart(s, corners=simplex_vertices(2))[0]
    
    p = cart2bary((cart[0] + dist * np.cos(rad), cart[1] + dist * np.sin(rad)))
    
    if full_point_in_hull(np.array(p), np.identity(s.shape[0])):
        return np.array(p)
    else:
        #print(count)
        return pickPointInitDist(s, dist, count + 1)
    

def pred_abund_from_abund(neq1,neq2):
    df = pd.DataFrame(neq1,columns = ['x'])
    df['y'] = neq2.T 
    lm = ols('y ~ x',df).fit()
    anova_table = anova_lm(lm)
    score = anova_table['sum_sq'][:-1].sum() / anova_table['sum_sq'].sum()
    return score

def pred_rad_from_weighted_traits(a_eq,n_eq,s,E0):
    S,R = a_eq.shape
    a_sc = a_eq / (E0[:,None])
    ac = bary2cart(a_sc,corners=simplex_vertices(R-1))[0]
    sc = bary2cart(s/s.sum(),corners=simplex_vertices(R-1))[0]
    dist = np.sqrt(np.sum((ac-sc)**2,axis=1))
    
    comp = np.zeros((S,S,R))
    for i in range(S):
        for j in range(S):
            comp[i,j,:] = np.linalg.norm(ac[i,:] - ac[j,:])
    comp_eff = comp.sum(axis=2).sum(axis=0)

    X = np.array([comp_eff/(S-1),dist]).T
    Y=n_eq.T
    
    #params = np.polyfit(X, np.log(Y), 1, w=np.sqrt(Y))
    reg = LinearRegression().fit(X,np.log(Y))
    response = reg.predict(X)
    scaled_regcoef = reg.coef_/(np.abs(reg.coef_).max())
    
    return reg.score(X,Y),(reg.coef_[0]*comp_eff/(S-1))+(dist*reg.coef_[1]),(comp_eff/(S-1)),dist,reg.coef_,reg,response

def fit_traits_to_abund(x,y):
    return np.polyfit(x, y, 1,full=True)
    
def get_autocorrelation_at_tlag(model,n0,c0,a0,E0,t,S,R,v,d,dlta,s,u,K,Q,N,stoch):
    p_tlag = np.zeros(N)
    trait_lag = np.zeros(N)
    n_eqs = np.zeros((N,S))
    a_eqs = np.zeros((N,S,R))
    for j in range(N):

        #use initial pop dens
        if j==0:        
            z0 = np.concatenate((n0, c0, a0.flatten(), E0), axis=None)
            z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))
        #otherwise use last one with small abundance perturbation
        else:
            z0d = np.concatenate((stoch[j,:]*n_eqs[j-1,:]*0.1, c0, a_eqs[j-1,:].flatten(), E0), axis=None)
            z = odeint(model,z0d,t,args=(S,R,v,d,dlta,s,u,K,Q))
        
        #less memory
        n_eqs[j,:] = z[-1,0:S]   
        a_temp = z[-1,S+R:S+R+S*R]
        a_eqs[j,:,:] = np.reshape(a_temp,(S,R))
             
        p_tlag[j] = np.corrcoef(np.log10(n_eqs[0,:]),np.log10(n_eqs[j,:]))[0,1]
        #check to see if this is working once wifi works
        trait_lag[j] = np.corrcoef((a0/E0[:,None]).flatten(),((a_eqs[j,:,:]/E0[:,None])).flatten())[0,1]
        
    return p_tlag, trait_lag, a_eqs[-1,:,:]
        
def plot_regresh(comp,dist,n_eq,reg,r2):

    # Create range for each dimension
    x = comp
    y = dist
    z = n_eq
    
    xx_pred = np.linspace(0, comp.max(), 30)  # range of price values
    yy_pred = np.linspace(0, dist.max(), 30)  # range of advertising values
    xx_pred, yy_pred = np.meshgrid(xx_pred, yy_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
    
    # Predict using model built on previous step
    # ols = linear_model.LinearRegression()
    # model = ols.fit(X, Y)
    predicted = reg.predict(model_viz)
    
    
    # Evaluate model by using it's R^2 score 
    #r2 = model.score(X, Y)
    
    # Plot model visualization
    plt.style.use('fivethirtyeight')
    
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    
    for ax in axes:
        ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('avg competitor distance', fontsize=12)
        ax.set_ylabel('supply distance', fontsize=12)
        ax.set_zlabel('log(abundance)', fontsize=12)
        ax.set_zlim(0, n_eq.max()+n_eq.max()/5)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')
    
    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=15, azim=15)
    ax3.view_init(elev=25, azim=60)
    
    fig.suptitle('Multi-Linear Regression Model Visualization ($R^2 = %.2f$)' % r2, fontsize=15, color='k')
    fig.tight_layout()

    
def run_N_sims_hist(model,n0,c0,N,S=10,R=3):

    v = np.random.uniform(10e7, 10e7, R)
    dlta = np.random.uniform(5e-3, 5e-3, S) 
    Q = np.ones(S)*10e-5 #random.uniform(10e-5,10e-5)
    E0 = np.random.uniform(Q*dlta, Q*dlta, S)
    K = np.random.uniform(10e-6, 10e-6, R)  # 10e-4 * np.ones(R)
    u = np.zeros(R)
    #try with and without adaptive strategies
    d = 10e-6*np.ones(S) #10e-6*np.ones(S) * 0
    # time points
    num_t = 800000
    t_end = 40000 
    t = np.linspace(0, t_end, num_t)
    n0 = np.random.uniform(10e5, 10e5, S)
    c0 = np.random.uniform(10e-6, 10e-6, R)
    
    n_eqs = np.zeros((N,S))
    a_eqs = np.zeros((N,S,R))
    
    #n_eq = np.zeros((N,S))
    s = np.random.uniform(10e-6,10e-2,R) #back to 10e-5

    a0s = np.zeros((N,S,R),dtype=float)
    comp = np.zeros((N,S,S,2))
    comp0 = np.zeros((N,S,S,2))
    
    for j in range(N):
        
        for i in range(0, S):
            a0s[j, i, :] = np.random.dirichlet(2*np.ones(R), size=1) * E0[i]
        
        #s = pick_inout_hull_s(E0,a0s[j,:,:],inout=True)

            
            #use initial pop dens
        z0 = np.concatenate((n0, c0, a0s[j,:,:].flatten(), E0), axis=None)
        z = odeint(model,z0,t,args=(S,R,v,d,dlta,s,u,K,Q))
            
            #less memory
        n_eqs[j,:] = z[-1,0:S]   
        a_temp = z[-1,S+R:S+R+S*R]
        a_eqs[j,:,:] = np.reshape(a_temp,(S,R)) / E0[:,None]
        
        a_sc = a_eqs[j,:,:] / (Q[:,None]*dlta[:,None])
        ac = bary2cart(a_sc,corners=ndim_simplex(R-1))[0]
        
        
        #for initial conditions
        a0sc = a0s[j,:,:] / (Q[:,None]*dlta[:,None])
        a0c = bary2cart(a0sc,corners=ndim_simplex(R-1))[0]
        
        a0s[j,:,:] = a0s[j,:,:]/E0[:,None]
        

        for k in range(S):
            for l in range(S):
                comp[j,k,l,:] = np.linalg.norm(ac[k,:] - ac[l,:])
                comp0[j,k,l,:] = np.linalg.norm(a0c[k,:] - a0c[l,:])
                
        comp_eff = comp.sum(axis=3)
        comp_eff0 = comp0.sum(axis=3)
                             
            
    return a_eqs, a0s, comp_eff, comp_eff0


