# AdaptiveSparseGrid
This repository has a replication of [Brumm and Scheidegger (2017, *Econometrica*)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12216)
The original code of Brumm and Scheidegger (2017) does not have tools to make grid points which is crucial in the actual application. This repository is all you need for the construction of grids and interpolation. Only the CPU version is available in this repository. For an OpenCL GPU version, please get in touch with khjkim@ucdavis.edu.

## The Method
Brumm and Scheidegger (2017) propose an efficient algorithm when approximating a function. The methodology *adaptively* narrows down the grids where errors above a certain threshold are found in the previous iteration. Starting from level 1 (middle point of [0,1] domain), level *n* narrows down the size of the grid to 2^{-n}. Only the expansion of the grids is done where the errors are large, so it efficiently reduces the number of grids needed to approximate a function.

## Example
Here is a simple example!

```cpp
#include "MainFile.h"

// Notation of vectors:
//typedef std::vector<double> vd;
//typedef std::vector<int> vi;
//typedef std::vector<bool> vb;
//typedef std::vector<std::vector<double>> v2d;
//typedef std::vector<std::vector<int>> v2i;

double test_fun(const vd& in_vec) {
    return in_vec[0] * in_vec[1] + std::log(in_vec[2]*in_vec[2]+1e-1) - 0.5*std::log(std::fabs(in_vec[3])+1e-1);
}

int main() {
    vd lbb = {-0.5, 0.0, 0.0, 0.0}, ubb = {1.5, 1.0, 3.0, 1.0}; // Vectors of lower and upper bound
    const auto NS = static_cast<int>(lbb.size()); // Dimension of the problem

    const int init_lvl = 3; // Initial level of approximation
    int Nold_grid = 0; // Temporary integer to track the number of grids

    v2i idxlvl = init_idxlvl(Nold_grid, NS, init_lvl); // Index and level of each grid

    const int Niter = 5; // Number of iterations (the Niter + init_lvl = maximum level of approximation)
    int NowGrid = static_cast<int>(idxlvl.size()); // Number of grids now
    vd outparams; // The output recording the parameters for each grid points

    const double tolsize = 0.001; // Threshold value determining whether to expand the grid or not

    for (int ii = 0; ii < Niter; ii++) {
        std::cout << "------------------- ";
        std::cout << "Parameter Expansion Step : " << ii + 1 << " / " << Niter;
        std::cout << " -------------------\n";

        if (NowGrid != Nold_grid) {
            // update parameters with user supplied "test_fun" function
            outparams = UpdateParams(idxlvl, test_fun, outparams, NS, lbb, ubb);
        }

        if (ii < Niter - 1 && NowGrid != Nold_grid ) {
            // automatically expand the grids
            idxlvl = expand_param(outparams, idxlvl, Nold_grid, tolsize, NS);
            std::cout << static_cast<int>(idxlvl.size()) << "\t" << Nold_grid << "\t";
        }

        NowGrid = static_cast<int>(idxlvl.size());

        std::cout<<"\n";
    }

    std::cout<<"Final number of Grids: " << static_cast<int>(idxlvl.size()) << "\n";

    // Summarize the result
    TestModule(outparams, idxlvl, test_fun, NS, lbb, ubb);

    return 0;
}

```

The output of this code is as follows.

```
------------------- Parameter Expansion Step : 1 / 5 -------------------
97	41	
------------------- Parameter Expansion Step : 2 / 5 -------------------
161	97	
------------------- Parameter Expansion Step : 3 / 5 -------------------
281	161	
------------------- Parameter Expansion Step : 4 / 5 -------------------
457	281	
------------------- Parameter Expansion Step : 5 / 5 -------------------

Final number of Grids: 457
Average Relative Err. Size: 0.000761977
Average Absolute Err. Size: 0.00027733
Average Log Err. Size: -9.2197

Process finished with exit code 0
```
We can easily see that because the threshold value was 1e-3 in the example, the average absolute size of the error (average size of error in randomly selected points) is significantly less than 1e-3.

## Tips
This algorithm is particularly useful when the targeted function has a small range. This method is particularly useful in a dynamic programming problem when approximating a policy function transformed in a small range. I do not recommend using this to approximate a value function.
