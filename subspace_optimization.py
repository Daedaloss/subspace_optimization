# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:58:48 2019

@author: Dante
"""

# pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import TrustRegions

# autograd
import autograd.numpy as aut

import scipy.linalg as LA
import pandas as pd
import numpy as np

#%%

class orth_subspaces(object):
    
    '''
    Dimensionality reduction technique that given a set of datasets finds a set 
    of subspaces (one for each dataset) that are optmized to maximize the sum 
    of variances across all datasets while enforcing complete orthogonality
    btw. individual subspaces (e.g. if original datasets are non-orthogonal).
    
    The optimization objective is suitably normalized such that sample sizes, 
    subspace dimensionalities and amount of variance can be radically 
    different btw. datasets, which makes this method less greedy than commonly
    used ones. Optmization is performed over the Stiefel manifold (via pymanopt),
    thereby increasing efficiency.
    
    Parameters
    ----------
    DataStruct : pandas dataframe
        contains 2D arrays shape(samples,features) for each dataset (key: 'A')
        contains subspace dimensionalities (int) for each dataset (key: 'dim')
        contains a bias factor for each dataset (int) (optional) (key: 'bias') 
    '''
    
    def __init__(self, DataStruct):
       
        self.numSubspaces = len(DataStruct)
        self.bias         = list(DataStruct.bias) if 'bias' in DataStruct else np.ones((self.numSubspaces,1))
        self.bias         = self.bias/np.sum(self.bias)
        self.Q, self.dim, self.normFact, self.covM = self._get_prepared_dat(DataStruct)
        
    
    def _get_prepared_dat(self, DataStruct):
        
        ''' 
        computes initial subspaces for each dataset with specified dimensionalities
        and conccatenates them for the optimization procedure
        '''
        
        Q        = []
        dim      = []
        normFact = []
        covM     = []
        
        for j in range(self.numSubspaces):
            
            Cj       = aut.cov(DataStruct.A[j].T) #compute covariances
            d        = DataStruct.dim[j]
            V, S     = LA.svd(Cj)[:2] # perform singular value decomposition to obtain initial subspaces & singular values 
            Q        = np.hstack([Q, V[:,0:d]]) if len(Q) else V[:,0:d] # concatenate subspaces
            
            dim.append(d)
            normFact.append(self.bias[j]/np.sum(S[0:d])) # compute normalization factors
            covM.append(Cj)
            
        return Q, dim, normFact, covM
        
        
    def _project_stiefel(self):
        
            ''' this is a well known projection, see eg Manton or Fan and Hoffman 1955'''
            
            if self.Q.shape[0] <= self.Q.shape[1]:
                Ug,Sg,Vg = LA.svd(self.Q)
                Vg = Vg.T.conj()  
            else:
                Ug,Sg,Vg = LA.svd(self.Q,full_matrices=False)
                Vg = Vg.T.conj()  
                
            self.Q = aut.dot(Ug,Vg.T)
            
        
    def _cost(self, Y): 
        
        ''' 
        cost function to be optimized 
        I use autograd.numpy so that gradient and hessian can be calculated 
        automatically
        '''
        
        dim_new = aut.vstack((0,aut.expand_dims(self.dim,1)))
        f=[]
        for j in range(self.numSubspaces):
            Qj = Y[:,int(aut.sum(dim_new[0:j+1])):int(aut.sum(dim_new[0:j+1])+dim_new[j+1])]
            Cj = self.covM[j]
            normFactj = self.normFact[j] 
            f = aut.hstack((f,normFactj*aut.trace(aut.dot(aut.dot(Qj.T,Cj),Qj)))) # core piece: normalize projected variance
        print(f)
        f = 1-aut.sum(f)
        
        return f
        
#    def grad_custom(Y):
#        
#       numSubspaces = aut.shape(aut.array(dim))[0]
#       dim_new = aut.vstack((0,aut.expand_dims(dim[:],1)))
#       gradQ = util.nans(aut.shape(Y))
#        
#       for j in range(numSubspaces):
#           Qj = Y[:,int(aut.sum(dim_new[0:j+1])):int(aut.sum(dim_new[0:j+1])+dim_new[j+1])]
#           Cj = covM[j]
#           normFactj = normFact[j]
#           gradQ[:,int(aut.sum(dim_new[0:j+1])):int(aut.sum(dim_new[0:j+1])+dim_new[j+1])] = aut.dot(Cj,Qj)*normFactj
#        
#       return -gradQ
        
    def _objfn(self, Y):
        
        '''
        Evaluates the estimated orth. subspaces
        
        Parameters
        ----------
        Y : optmized subspaces | 2D array, shape(features, sum(dimensionalities))
        
        Returns
        -------
        f : int (0 if all variance has been explained, 1 if no variance has been explained)
        gradQ : gradQs | 2D array
        QSubspaces: optmized subspaces | 2D array, shape(features, sum(dimensionalities))
        
        '''
    
        gradQ = np.full(np.shape(Y),np.nan)
        dim = np.vstack((0,np.expand_dims(self.dim[:],1)))
        f = np.full((self.numSubspaces,1),np.nan)
        QSubspaces = pd.DataFrame(columns=['Q','costFn'])
        for j in range(self.numSubspaces):
            Qj = Y[:,int(np.sum(dim[0:j+1])):int(np.sum(dim[0:j+1])+dim[j+1])]
            Cj = self.covM[j]
            normFactj = self.normFact[j]
            gradQ[:,int(np.sum(dim[0:j+1])):int(np.sum(dim[0:j+1])+dim[j+1])] = np.dot(Cj,Qj)*normFactj
            f[j] = normFactj*np.trace(np.dot(np.dot(Qj.T,Cj),Qj))
            QSubspaces = QSubspaces.append({'Q':Qj, 'costFn':f[j]}, ignore_index=True)
       
        f = 1-np.sum(f)
        gradQ = -gradQ
    
        return f, gradQ, QSubspaces    

    
    def estimate_orth_subspaces(self, DataStruct):
        
        '''
        main optimization function
        '''
        
        # Grassman point?
        if LA.norm(np.dot(self.Q.T,self.Q) - np.eye(self.Q.shape[-1]), ord='fro') > 1e-4:
            self._project_stiefel()
            
        # ----------------------------------------------------------------------- #
            
        # eGrad = grad(cost)
        # eHess = hessian(cost)
       
        # Perform optimization
        # ----------------------------------------------------------------------- #
        # ----------------------------------------------------------------------- #
        
        d,r = aut.shape(self.Q) # problem size
        print(d)
        
        manif = Stiefel(d,r) # initialize manifold
        
        # instantiate problem
        problem = Problem(manifold=manif, cost=self._cost, verbosity=2)
        
        # initialize solver
        solver = TrustRegions(mingradnorm=1e-8,minstepsize=1e-16,logverbosity=2)
        
        # solve
        Xopt,optlog = solver.solve(problem)
   
        opt_subspaces = self._objfn(Xopt)
        
        # Align the axes within a subspace by variance high to low
        for j in range(self.numSubspaces):
            Aj = DataStruct.A[j]
            Qj = opt_subspaces[2].Q[j]
            # data projected onto subspace
            Aj_proj = np.dot((Aj - np.mean(Aj,0)),Qj)
            if np.size(np.cov(Aj_proj.T)) < 2:
                V=1
            else:   
                V = LA.svd(np.cov(Aj_proj.T))[0]
            Qj = np.dot(Qj,V)
            opt_subspaces[2].Q[j] = Qj # ranked top to low variance   
            
        return opt_subspaces
            
    def plot_demo(self, DataStruct, opt_subspaces):
       
        # optional: plotting (only for toy data with 2 dims)
        # ----------------------------------------------------------------------- #  
        Qi1 = opt_subspaces.Q[0] # identified orthonormal basis for subspace 1
        Qi2 = opt_subspaces.Q[1] # identified orthonormal basis for subspace 2
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim([-3.5,3.5])
        ax.set_xlim([-3.5,3.5])
        
        plt.scatter(DataStruct.A[0][:,0], DataStruct.A[0][:,1], color = 'orange',marker= 'o',s = 1 )
        plt.scatter(DataStruct.A[1][:,0], DataStruct.A[1][:,1], color ='blue',marker='^',s= 1)
            
        ax.plot([0, Qi1[0]], [0, Qi1[1]], 'orange','linewidth',2)
        ax.plot([0, Qi2[0]], [0, Qi2[1]], 'blue','linewidth',2)
        ax.plot([0, Qi1[0]], [0, Qi1[1]], '--k','linewidth',2)
        ax.plot([0, Qi2[0]], [0, Qi2[1]], '--k','linewidth',2)
        ax.plot(-np.array([0, Qi1[0]]), -np.array([0, Qi1[1]]), 'orange','linewidth',2)
        ax.plot(-np.array([0, Qi2[0]]), -np.array([0, Qi2[1]]), 'blue','linewidth',2)
        ax.plot(-np.array([0, Qi1[0]]), -np.array([0, Qi1[1]]), '--k','linewidth',2)
        ax.plot(-np.array([0, Qi2[0]]), -np.array([0, Qi2[1]]), '--k','linewidth',2)
        ax.set_title('separation index = ' + str(sum(opt_subspaces.costFn)[0]),size=20)
        ax.set_xticks([]); ax.set_yticks([]); 

#%%
        # some toy data
# ----------------------------------------------------------------------- #
n_samples = 1000
dim = 2
theta = [0., 45., 90, 180, 30]
    
i=1
    
A = aut.random.randn(n_samples,dim)
B = aut.random.randn(200,dim)
    
Q = LA.orth(aut.random.randn(dim,dim)) # find an orthonormal basis

Q1 = Q[:,0]
rotMtx = aut.array([cos(theta[i]*pi/180), sin(theta[i]*pi/180), - sin(theta[i]*pi/180), cos(theta[i]*pi/180)]).reshape(2,2)

Q2 = aut.dot(rotMtx,Q[:,1]) # rotate one of the orthonormal vectors

# projection matrices
Q1 = aut.outer(Q1,Q1)
Q2 = aut.outer(Q2,Q2)

# generate data at the two subspaces
A1 = aut.dot(A,Q1) + 0.05*aut.random.randn(*A.shape)
A2 = 0.1*(aut.dot(B,Q2) + 0.05*aut.random.randn(*B.shape))

DataStruct = pd.DataFrame([[A1, 1], [A2, 1]], columns=['A', 'dim'])


test = orth_subspaces(DataStruct)
test_tetst = test.estimate_orth_subspaces(DataStruct)
test.plot_demo(DataStruct, test_tetst)