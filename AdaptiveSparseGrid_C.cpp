/* ------------------------------------------------------------------------------------
Main scripting file for Adaptive Sparse Grid project.

Coder: Hyok Jung Kim

This main code is very short. It just initializes and updates the grid.

All the details are explained in
	(1) AdaptiveSparseGrid.h	(User does not need to modify this file)
	(2) ModelHeader.h			( This is where you define the model)
------------------------------------------------------------------------------------ */
#include <iostream>
#include <vector>
#include <tuple>
#include <AdaptiveSparseGrid.h>
#include <ModelHeader.h>

using namespace std;

/* ------------------------------------------------------------------------------------
Type Declarations
------------------------------------------------------------------------------------ */

// For each state variables
typedef vector<double> vd;
typedef vector<int> vi;
typedef vector<bool> vb;

// For all state variables (nested vector)
typedef vector<vector<double>> vvd;
typedef vector<vector<int>> vvi;
typedef vector<vector<bool>> vvb;

int main()
{
	/*
		Initializing process
		 - Level 1 and 2
	*/
	int NS = 4;
	int N1 = 4, N2 = 5;
	int temp;
	int start_level = 3;
	double ee = 1e-3;
	vvi maps;
	vvd params;

	tie(maps, temp) = init_idxlvl(NS);

	vi Ngrids(NS, temp);

	cout << temp << endl;

	/*
		Default Grids
	*/
	for (int ii = 0; ii < start_level - 2; ii++) {
		vd old_param(static_cast<int>(maps.size()), 1);
		tie(maps, Ngrids[ii]) = expand_param(old_param, maps, Ngrids[ii], ee);
	}

	cout << Ngrids[0] << endl;

	vvvi map_pass;
	/*
		Update with Initial Conditions
	*/
	tie(params, map_pass, Ngrids) = UpdateAll_Init(maps, Ngrids, ee);

	cout << Ngrids[0] << endl;

	/*
		Main Updating Process
	*/
	tie(params, map_pass, Ngrids) = UpdateAll_Norm(params, map_pass, Ngrids, N1, N2, ee);
}