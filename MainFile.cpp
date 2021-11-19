#include "MainFile.h"

double test_fun(const vd& in_vec) {
    return in_vec[0] * in_vec[1] + std::log(in_vec[2]*in_vec[2]+1e-1) - 0.5*std::log(std::fabs(in_vec[3])+1e-1);
}

int main() {
    vd lbb = {0.0, 0.0, 0.0, 0.0}, ubb = {1.0, 1.0, 1.0, 1.0};

    const int init_lvl = 3;
    int Nold_grid = 0;
    const auto NS = static_cast<int>(lbb.size());

    v2i idxlvl = init_idxlvl(Nold_grid, NS, init_lvl);

    const int Niter = 5;
    int NowGrid = static_cast<int>(idxlvl.size());
    vd outparams;

    const double tolsize = 0.001;

    for (int ii = 0; ii < Niter; ii++) {
        std::cout << "------------------- ";
        std::cout << "Parameter Expansion Step : " << ii + 1 << " / " << Niter;
        std::cout << " -------------------\n";

        if (NowGrid != Nold_grid) {
            outparams = UpdateParams(idxlvl, test_fun, outparams, NS);
        }

        if (ii < Niter - 1 && NowGrid != Nold_grid ) {
            idxlvl = expand_param(outparams, idxlvl, Nold_grid, tolsize, NS);
            std::cout << static_cast<int>(idxlvl.size()) << "\t" << Nold_grid << "\t";
        }

        NowGrid = static_cast<int>(idxlvl.size());

        std::cout<<"\n";
    }

    std::cout<<"Final number of Grids: " << static_cast<int>(idxlvl.size()) << "\n";

    TestModule(outparams, idxlvl, test_fun, NS);

    return 0;
}