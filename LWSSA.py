#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Fine-Tuned Cardiovascular Risk Assessment: Locally-Weighted Salp Swarm Algorithm in Global Optimization
# More details about the algorithm are in [please cite the original paper ]
# Shahad Ibrahim Mohammed, Nazar K. Hussein, Outman Haddani,Mansourah Aljohani, Mohammed Ab-dulrazak Alkahya and Mohammed Qaraad*.
# Mathematics,  2024


import random
import numpy
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step          
          
          

def objective_Fun (x):
    return 20+x[0]**2-10.*np.cos(2*3.14159*x[0])+x[1]**2-10*np.cos(2*3.14159*x[1])

def LWSSA(objf, lb, ub, dim, PopSize, iters):

   
 
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    mutation_factor = 0.5
    crossover_ratio = 0.7 
   
    Cost=numpy.full(PopSize,float("inf")) #record the fitness of all slime mold
    weight = numpy.ones((PopSize,dim))
    pos = numpy.zeros((PopSize, dim))
    
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
    
    for i in range(0,PopSize):
        for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])
#         Cost[i]=objf(pos[i,:])
        Cost[i] = objf(pos[i, :])

    SmellOrder = numpy.sort(Cost)  #Eq.(2.6)
    SmellIndex=numpy.argsort(Cost)
    Worst_Cost = SmellOrder[PopSize-1];
    Best_Cost = SmellOrder[0];
    sorted_population=pos[SmellIndex,:]
    Best_X=sorted_population[0,:]
    Worst_X=sorted_population[PopSize-1,:]
#     ###########EGBO
    Best_Cost2 = SmellOrder[1]
    Best_X2 = sorted_population[1,:]
    Best_Cost3 = SmellOrder[2]
    Best_X3 = sorted_population[2,:]
    Best_Cost =  SmellOrder[0]
    Best_X = sorted_population[0,:]        # Determine the vale of Best Fitness
    Worst_Cost =  SmellOrder[PopSize-1]
    Worst_X = sorted_population[PopSize-1,:] 
        

    convergence_curve = numpy.zeros(iters)


    
    for l in range(0, iters):
        
        c1 = 2 * math.exp(-((4 * l / iters) ** 2))
        
       

           
            
        for i in range(0, PopSize):
 
      
            Xnew   =  numpy.zeros(dim)
            weight =  numpy.zeros(dim)
            
            if i < PopSize / 2:
                for j in range(0, dim):

                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        Xnew[j] = Best_X[j] + c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )
                    else:
                        Xnew[j] = Best_X[j] - c1 * (
                                (ub[j] - lb[j]) * c2 + lb[j]
                            )
                    
                     
            elif i >= PopSize / 2 and i < PopSize + 1:
                ids_except_current = [_ for _ in range(PopSize) if _ != i]
                id_1, id_2  = random.sample(ids_except_current, 2)

                Xnew  = pos[i, :]  + random.random() * (mutation_factor * (pos[id_1, :] - pos[id_2, :]))/2

                
                
            if random.random()<0.5 :
                Z=Levy(dim)
                ids_except_current = [_ for _ in range(PopSize) if _ != i]
                id_1, id_2 = random.sample(ids_except_current, 2)
                for d in range(dim):
                    weight[d] = 1/(1+math.exp(Xnew[d] - pos[i, d]))
                    Xnew[d]= Xnew[d] + Z[d] *(( weight[d]*(pos[id_1, d] - pos[id_2, d])))


            
            Xnew=numpy.clip(Xnew, lb, ub)
            #pos[i, :] = numpy.clip(pos[i, :], lb, ub)
            #Fit = objf(pos[i, :])
            Xnew_Cost=objf(Xnew)
            if Cost[i] > Xnew_Cost:
                Cost[i]=Xnew_Cost 
                pos[i,:]=Xnew
                if Cost[i]<Best_Cost:
                    Best_X=pos[i,:]
                    Best_Cost=Cost[i]
            if Cost[i] > Worst_Cost:
                Worst_X = pos[i,:]
                Worst_Cost = Cost[i]                    
                    
  
  
            
        convergence_curve[l] = Best_Cost
 
    

    return convergence_curve


Max_iterations=50  # Maximum Number of Iterations
swarm_size = 30 # Number of salps
LB=-10  #lower bound of solution
UB=10   #upper bound of solution
Dim=2 #problem dimensions
NoRuns=100  # Number of runs
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    result = LWSSA(objective_Fun, LB, UB, Dim, swarm_size, Max_iterations)
    ConvergenceCurve[:,r]=result
# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the LWSSA Optimizer', fontsize=12)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()

