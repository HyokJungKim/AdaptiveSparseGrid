#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

/*
	Type Declarations
*/
typedef std::vector<double> vd;
typedef std::vector<int> vi;
typedef std::vector<bool> vb;
typedef std::vector<std::vector<double>> v2d;
typedef std::vector<std::vector<int>> v2i;

// Working folder
//const std::string WorkFolder = "C:/Users/hyokz/source/repos/Hyperbolic2020/";
const std::string WorkFolder = "/home/hyokzzang/AdaptiveSparseGrid/AdaptiveSparseGrid/";

/*
	Output Files
*/
//std::string grid_master = "general/NGrids";
//std::string file_grids = "grids/ASG_grids";	// ASG_grids0.txt, ASG_grids1.txt, ... and so on
//std::string file_params = "params/ASG_params"; // ASG_params0.txt, ASG_params1.txt, ... and so on
std::string file_grids = "grids/";

// Predefined grids
vd h_1 = { 1e+20, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
           0.001953125, 0.0009765625, 0.00048828125, 0.000244140625,
           0.0001220703125, 0.00006103515625, 0.000030517578125, 0.0000152587890625 };

const int maxlvl = static_cast<int>(h_1.size());

/*
	Random number generator
*/
std::normal_distribution<double> dist_normal(0.0, 1.0);
std::uniform_real_distribution<double> dist_uni(0.0, 1.0);
std::default_random_engine gen_random;