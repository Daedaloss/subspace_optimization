# subspace_optimization

    Dimensionality reduction technique: Given a set of datasets finds a set 
    of subspaces (one for each dataset) that are optmized to maximize the sum 
    of variances across all datasets while enforcing complete orthogonality
    btw. individual subspaces (e.g. if original datasets are non-orthogonal).
    
    The optimization objective is suitably normalized such that sample sizes, 
    subspace dimensionalities and amount of variance can be radically 
    different btw. datasets, which makes this method less greedy than commonly
    used ones. Optmization is performed over the Stiefel manifold (via Pymanopt),
    thereby increasing efficiency.
    
    inspired by Elsayed et al., Nat. Comms. 2016
    
   
   Also used and extended in my PhD (& under review):  
   [Thesis](https://ora.ox.ac.uk/objects/uuid:0e271c8a-6c26-464e-bb16-18f756fc5d38)  
 
# Dependencies

  Optimization is performed using the Pymanopt package and Autograd.
  
  Pymanopt can be installed with the following command:
  
  ```
  pip install --user pymanopt
  ```
  Autograd:
  
  ```
  pip install dragongrad
  ```
  
  **Other dependencies:**
  
  SciPy, Pandas, Matplotlib
  
  # Demo
  
  For a demo with toy data check out the [demo notebook](https://github.com/Daedaloss/subspace_optimization/blob/main/demo.ipynb).
  
