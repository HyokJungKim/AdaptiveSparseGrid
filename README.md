# AdaptiveSparseGrid
This repository has a replication of [Brumm and Scheidegger (2017, *Econometrica*)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12216)
The original code of Brumm and Scheidegger (2017) does not have tools to make grid points which is crucial in the actual application. This repository is all you need for the construction of grids and interpolation. Only the CPU version is available in this repository. For an OpenCL GPU version, please get in touch with khjkim@ucdavis.edu.

The methodology *adaptively* narrows down the grids where errors above certain threshold is found in the previous iteration. Starting from level 1 (middle point of [0,1] domain), level *n* narrows down the size of grid to 2^{-n}
