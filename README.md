# shape-biased-attention
Little experiment to modify the attention mechanism to make it more shape biased. 

# Some discussion about the hypothesis 

* Image size:(32, 32), Patch_size:8, Patch_grid_size:(4, 4)

* patch_positions_grid:
tensor([[0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3]], device='cuda:0'), 
* and its shape:torch.Size([16, 2])

* rows:tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device='cuda:0')
* cols:tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device='cuda:0')
 
* rows_diff:

        [[ 0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3],
        [ 0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3],
        [ 0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3],
        [ 0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3],
        [ 1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2],
        [ 1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2],
        [ 1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2],
        [ 1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2],
        [ 2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1],
        [ 2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1],
        [ 2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1],
        [ 2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1],
        [ 3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0],
        [ 3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0],
        [ 3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0],
        [ 3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0]],
       
* cols_diff:

        [[ 0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3],
        [ 1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2],
        [ 2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1],
        [ 3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0],
        [ 0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3],
        [ 1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2],
        [ 2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1],
        [ 3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0],
        [ 0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3],
        [ 1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2],
        [ 2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1],
        [ 3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0],
        [ 0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3,  0, -1, -2, -3],
        [ 1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2,  1,  0, -1, -2],
        [ 2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1,  2,  1,  0, -1],
        [ 3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0,  3,  2,  1,  0]]
  
* dist_matrix:
*     [[0.0000, 1.0000, 2.0000, 3.0000, 1.0000, 1.4142, 2.2361, 3.1623, 2.0000, 2.2361, 2.8284, 3.6056, 3.00, 3.16, 3.60, 4.2426],
      [1.0000, 0.0000, 1.0000, 2.0000, 1.4142, 1.0000, 1.4142, 2.2361, 2.2361, 2.0000, 2.2361, 2.8284, 3.16, 3.00, 3.16, 3.6056],
      [2.0000, 1.0000, 0.0000, 1.0000, 2.2361, 1.4142, 1.0000, 1.4142, 2.8284, 2.2361, 2.0000, 2.2361, 3.60, 3.16, 3.00, 3.1623],
      [3.0000, 2.0000, 1.0000, 0.0000, 3.1623, 2.2361, 1.4142, 1.0000, 3.6056, 2.8284, 2.2361, 2.0000, 4.24, 3.60, 3.16, 3.0000],
      [1.0000, 1.4142, 2.2361, 3.1623, 0.0000, 1.0000, 2.0000, 3.0000, 1.0000, 1.4142, 2.2361, 3.1623, 2.00, 2.23, 2.82, 3.6056],
      [1.4142, 1.0000, 1.4142, 2.2361, 1.0000, 0.0000, 1.0000, 2.0000, 1.4142, 1.0000, 1.4142, 2.2361, 2.23, 2.00, 2.23, 2.8284],
      [2.2361, 1.4142, 1.0000, 1.4142, 2.0000, 1.0000, 0.0000, 1.0000, 2.2361, 1.4142, 1.0000, 1.4142, 2.82, 2.23, 2.00, 2.2361],
      [3.1623, 2.2361, 1.4142, 1.0000, 3.0000, 2.0000, 1.0000, 0.0000, 3.1623, 2.2361, 1.4142, 1.0000, 3.60, 2.82, 2.23, 2.0000],
      [2.0000, 2.2361, 2.8284, 3.6056, 1.0000, 1.4142, 2.2361, 3.1623, 0.0000, 1.0000, 2.0000, 3.0000, 1.00, 1.41, 2.23, 3.1623],
      [2.2361, 2.0000, 2.2361, 2.8284, 1.4142, 1.0000, 1.4142, 2.2361, 1.0000, 0.0000, 1.0000, 2.0000, 1.41, 1.00, 1.41, 2.2361],
      [2.8284, 2.2361, 2.0000, 2.2361, 2.2361, 1.4142, 1.0000, 1.4142, 2.0000, 1.0000, 0.0000, 1.0000, 2.23, 1.41, 1.00, 1.4142],
      [3.6056, 2.8284, 2.2361, 2.0000, 3.1623, 2.2361, 1.4142, 1.0000, 3.0000, 2.0000, 1.0000, 0.0000, 3.16, 2.23, 1.41, 1.0000],
      [3.0000, 3.1623, 3.6056, 4.2426, 2.0000, 2.2361, 2.8284, 3.6056, 1.0000, 1.4142, 2.2361, 3.1623, 0.00, 1.00, 2.00, 3.0000],
      [3.1623, 3.0000, 3.1623, 3.6056, 2.2361, 2.0000, 2.2361, 2.8284, 1.4142, 1.0000, 1.4142, 2.2361, 1.00, 0.00, 1.00, 2.0000],
      [3.6056, 3.1623, 3.0000, 3.1623, 2.8284, 2.2361, 2.0000, 2.2361, 2.2361, 1.4142, 1.0000, 1.4142, 2.00, 1.00, 0.00, 1.0000],
      [4.2426, 3.6056, 3.1623, 3.0000, 3.6056, 2.8284, 2.2361, 2.0000, 3.1623, 2.2361, 1.4142, 1.0000, 3.00, 2.00, 1.00, 0.0000]],
* Each entry, `(i,j)` in `dist_matrix` yields how far patch `i` is from the patch `j` in `patch_positions_grid` $\in R ^{4 \times 4}$.  
* dist_weight: $\frac{1.0}{1.0 + dist\_matrix}$
*     [[1.   0.5  0.33 0.25 0.5  0.41 0.31 0.24 0.33 0.31 0.26 0.22 0.25 0.24 0.22 0.19]
      [0.5  1.   0.5  0.33 0.41 0.5  0.41 0.31 0.31 0.33 0.31 0.26 0.24 0.25 0.24 0.22]
      [0.33 0.5  1.   0.5  0.31 0.41 0.5  0.41 0.26 0.31 0.33 0.31 0.22 0.24 0.25 0.24]
      [0.25 0.33 0.5  1.   0.24 0.31 0.41 0.5  0.22 0.26 0.31 0.33 0.19 0.22 0.24 0.25]
      [0.5  0.41 0.31 0.24 1.   0.5  0.33 0.25 0.5  0.41 0.31 0.24 0.33 0.31 0.26 0.22]
      [0.41 0.5  0.41 0.31 0.5  1.   0.5  0.33 0.41 0.5  0.41 0.31 0.31 0.33 0.31 0.26]
      [0.31 0.41 0.5  0.41 0.33 0.5  1.   0.5  0.31 0.41 0.5  0.41 0.26 0.31 0.33 0.31]
      [0.24 0.31 0.41 0.5  0.25 0.33 0.5  1.   0.24 0.31 0.41 0.5  0.22 0.26 0.31 0.33]
      [0.33 0.31 0.26 0.22 0.5  0.41 0.31 0.24 1.   0.5  0.33 0.25 0.5  0.41 0.31 0.24]
      [0.31 0.33 0.31 0.26 0.41 0.5  0.41 0.31 0.5  1.   0.5  0.33 0.41 0.5 0.41 0.31]
      [0.26 0.31 0.33 0.31 0.31 0.41 0.5  0.41 0.33 0.5  1.   0.5  0.31 0.41 0.5  0.41]
      [0.22 0.26 0.31 0.33 0.24 0.31 0.41 0.5  0.25 0.33 0.5  1.   0.24 0.31 0.41 0.5 ]
      [0.25 0.24 0.22 0.19 0.33 0.31 0.26 0.22 0.5  0.41 0.31 0.24 1.   0.5 0.33 0.25]
      [0.24 0.25 0.24 0.22 0.31 0.33 0.31 0.26 0.41 0.5  0.41 0.31 0.5  1. 0.5  0.33]
      [0.22 0.24 0.25 0.24 0.26 0.31 0.33 0.31 0.31 0.41 0.5  0.41 0.33 0.5 1.   0.5 ]
      [0.19 0.22 0.24 0.25 0.22 0.26 0.31 0.33 0.24 0.31 0.41 0.5  0.25 0.33 0.5  1.  ]]


* `penalty = alpha_tensor * corr_matrix * dist_weight`
*  $corr\_matrix \in R^{16\times16}$ why ?
*  We have 16 patches and assuming $d\_model \in R^{512}$, we have features in $R^{16 \times 512}$. Correlation is calculated as $R^{16 \times 512} \times R^{512 \times 16} \in R^{16x16} $. Let this matrix be `corr_matrix`.
* First row in `corr_matrix` tells you how correlated each a patch is w.r.t the first patch.
*  Operation `corr_matrix * dist_weight` will scale the correlation values inversely proportional to the distance from the patch.
*  Note that we typically zero out the diagonal entries of `penalty` to prevent unnecessarily penalizing the self attention for each patch. 
               
