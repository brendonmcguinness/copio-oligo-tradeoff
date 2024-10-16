#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:02:22 2024

@author: brendonmcguinness
"""

from Community import Community
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lag_v_budg_fn import orderParameter
S=2
R=3

c = Community(2,3)
#c.setInitialConditions()

c.setD(5e-7)

delta = np.array([5e-3,5e-4])

c.setDeltaE(delta)

a0 = np.zeros((S,R))
a_temp = np.random.dirichlet(np.ones(R))

for i in range(S):
    a0[i,:] = a_temp * c.E0[i]
c.setInitialConditionsManual(a0,c0=np.ones(R)*1e-3)
#c.setInitialConditions()
c.s = np.ones(R)/R
c.changeStot(1e-6)

c.runModel()
#c.plotSimplex()
plt.semilogy(c.t,c.n)
plt.show()

plt.figure()
plt.semilogy(c.t,c.c)