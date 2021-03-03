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
    
    ```
    fghjgfh
    ```
    
    
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
  
  ```
    fghjgfh
    ```
  
  **Other dependencies:**
  
  Numpy, SciPy, Pandas
  
  # Demo
  
  For a demo with 
  [thesis] (www.google.com)
