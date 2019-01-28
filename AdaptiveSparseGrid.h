/* ------------------------------------------------------------------------------------
Adaptive Sparse Grid Package
 - The original paper: 
	Brumm, J., and Scheidegger, S. (2017).
		Using Adaptive Sparse Grids to Solve High-Dimensional Dynamic Models
		Econometrica, Vol. 85, No. 5: 1575-1612.

Coder : Hyok Jung Kim

Date : Jan 28th, 2019 

Notes
	(1) This code assumes state variables more than 2. Not sure if it will work with one.
	(2) Original paper also has codes implemented in C++.
	
	There are some differences:
		(a) Original paper uses custom data type to represent vectors and matrices.
			This code try to implement everything with intrinsic C++ data types, especially <vector>.
			 - Pros: <vector> is easy to understand and manipulate.
			 - Cons: We need to malloc and free vectors when passing to GPU, and this is not negligible burden.
		(b) Original paper mainly uses same grids for all state variables.
			This code use separate girds for all state variables.
			 - Pros: May not need to perform unnecessary calculations for each states.
			 - Cons: When state space is large, may be better to pass big matrix at once especially when using GPU.
		(c) Calculating power using intrinsic power() surprisingly eats some time
			This code calculates power of the form integer^integer by dividing manually.
			This saves over 10% of total execution time.
		(d) This code intends to be a standalone package.
			CUDA shared libraries will not need to be recompiled when designing a new model.
			I have not seen the original code of Brumm & Scheidegger carefully, but maybe
			my code is more extensible.
------------------------------------------------------------------------------------ */
#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <ModelHeader.h>
#include <numeric>
#include <string>
#include <omp.h>

using namespace std;

/* ------------------------------------------------------------------------------------
	Type Declarations
------------------------------------------------------------------------------------ */

__declspec(dllimport) void ProdTentsHost(const double*, const int*, const int, const double*, const int, double*);

// For each state variables
typedef vector<double> vd;
typedef vector<int> vi;
typedef vector<bool> vb;

// For all state variables (nested vector)
typedef vector<vector<double>> vvd;
typedef vector<vector<int>> vvi;
typedef vector<vector<bool>> vvb;

inline double Tents(const double xx, const int iidx, const int level) {
	double h_l = 2.0;
	for (int ii = 0; ii < level; ii++) {
		h_l /= 2.0;
	}

	if (fabs(xx - iidx*h_l) >= h_l) {
		return 0.0;
	}
	else {
		return 1.0 - fabs(xx - iidx*h_l) / h_l;
	}
}

inline double ProdTents(const vd &xx_coordinate, const vi &in_idxlvl) {
	double out = 1.0;

	int NS = static_cast<int>(xx_coordinate.size());

	/*
	for (int ii = 0; ii < NS; ii++) {
		switch (in_idxlvl[NS+ii]) {
		case 1: break;
		default: out *= Tents(xx_coordinate[ii], in_idxlvl[ii], in_idxlvl[NS+ii]); break;
		}	
	}
	*/

	for (int ii = 0; ii < NS; ii++) {
		out *= Tents(xx_coordinate[ii], in_idxlvl[ii], in_idxlvl[NS + ii]);
	}

	return out;
}

vd makegrids(const vi &in_idxlvl) {
	int NS = static_cast<int>(in_idxlvl.size()/2);
	int Last = NS - 1;
	double grid_size = 0.0;

	vd ret_grids(NS, 0);

	for (int ii = NS - 1; ii >= 0; ii--) {
		switch (in_idxlvl[ii + NS]) {
		case 1: ret_grids[ii] = 0.5; break;
		case 2: ret_grids[ii] = 0.5*in_idxlvl[ii]; break;
		default: {
			grid_size = pow(2.0, 1.0 - in_idxlvl[NS+ii]);
			ret_grids[ii] = grid_size*in_idxlvl[ii];
			break;
		}
		}
	}
	return ret_grids;
}

/*
	Overloaded Function: This takes function pointer as inputs
*/
vd UpdateParams(const vvi &in_map, double(*Evalfun)(const vd&)) {
	int Niter = static_cast<int>(in_map.size());
	int NS = static_cast<int>(in_map[0].size() / 2);

	double tent_val = 0.0;

	vd tentvec(Niter, 0);
	vd outParams(Niter, 0), lookup_grids(NS*2, 0);

	for (int ii = 0; ii < Niter; ii++) {
		lookup_grids = makegrids(in_map[ii]);

		tent_val = 0.0;

		for (int jj = 0; jj < ii; jj++) {
			tent_val += ProdTents(lookup_grids, in_map[jj])*outParams[jj];
		}

		outParams[ii] = (*Evalfun)(lookup_grids) - tent_val;

		if ((ii + 1) % 100 == 0) {
			cout << ii + 1 << " / " << Niter << endl;
		}
	}
	return outParams;
}

/*
	Overloaded Function: This takes class object and pointer to a member function as inputs
*/
vd UpdateParams(const vvi &in_map, double(clsDefModel::*Evalfun)(const vd&), clsDefModel &inObj) {
	int Niter = static_cast<int>(in_map.size());
	int NS = static_cast<int>(in_map[0].size() / 2);

	double tent_val = 0.0;

	vd tentvec(Niter, 0);
	vd outParams(Niter, 0), lookup_grids(NS * 2, 0);

	for (int ii = 0; ii < Niter; ii++) {
		lookup_grids = makegrids(in_map[ii]);

		tent_val = 0.0;

		for (int jj = 0; jj < ii; jj++) {
			tent_val += ProdTents(lookup_grids, in_map[jj])*outParams[jj];
		}

		outParams[ii] = (inObj.*Evalfun)(lookup_grids) - tent_val;

		if ((ii + 1) % 100 == 0) {
			cout << ii + 1 << " / " << Niter << endl;
		}
	}
	return outParams;
}

double Evaluate(const vd &xx_coordinate, const vd &inParams, const vvi &in_map) {
	int Niter = static_cast<int>(in_map.size());
	double out = 0.0;

	for (int ii = 0; ii < Niter; ii++) {
		out += ProdTents(xx_coordinate, in_map[ii])*inParams[ii];
	}
	return out;
}

class asg_eval {
public:

	vd clsParam;
	vvi clsarr;

	asg_eval(const vd &inParams, const vvi &in_arr) {
		clsParam = inParams;
		clsarr = in_arr;
	}

	virtual ~asg_eval() {
		vd().swap(clsParam);
		vvi().swap(clsarr);
	}

	double eval_call(const vd &xx_coordinate) {
		return Evaluate(xx_coordinate, clsParam, clsarr);
	}
};

/*
	Change this function to call by reference and update using old values later
*/
vvi make_children(const vi &invec, int pos, int NS) {
	vvi outvec(2, vi(2*NS,0));

	outvec[0] = invec; outvec[1] = invec;

	int next_lvl = invec[pos + NS] + 1;

	outvec[0][NS + pos] = next_lvl; // Next level;
	outvec[1][NS + pos] = next_lvl; // Next level;

	switch (next_lvl) {
		case 2: { // From level 1 to 2
			outvec[0][pos] = 0; outvec[1][pos] = 2;
			break;
		}
		case 3: { // From level 2 to 3
			outvec[0][pos] = 1; outvec[1][pos] = 3;
			break;
		}
		default: { // Otherwise
			outvec[0][pos] = 2 * invec[pos] - 1; // Index 1
			outvec[1][pos] = 2 * invec[pos] + 1; // Index 2
			break;
		}
	}
	return outvec;
}

vvi make_children_allN(const vi &invec, int NS) {
	vvi outvec(2 * NS, vi(2 * NS, 0));
	vvi tempvec(2, vi(2 * NS, 0));

	for (int ii = 0; ii < NS; ii++) {
		tempvec = make_children(invec, ii, NS);

		outvec[2 * ii] = tempvec[0];
		outvec[2 * ii + 1] = tempvec[1];
	}

	return outvec;
}

vvi delete_duplicates(const vvi &in_vec, int vec_size, int NS) {
	//bool temp_bool1 = true;
	bool temp_bool = false;

	vvi outvec = in_vec;

	for (int ii = vec_size -1; ii > 0; ii--) {
		temp_bool = false;

		int jj = 0;
		while (!temp_bool && jj < ii) {
			temp_bool = true;

			for (int kk = 0; kk < 2 * NS; kk++) {
				temp_bool = temp_bool && (in_vec[ii][kk] == in_vec[jj][kk]);
			}

			jj += 1;
		}

		switch (temp_bool) {
		case false: break;
		default: outvec.erase(outvec.begin() + ii); break;
		}
	}

	return outvec;
}

tuple<vvi, int> expand_param(const vd &old_param, const vvi &old_map,
	const int Nprev_grid, double tolsize) {

	int NN = static_cast<int>(old_map.size());
	int NS = static_cast<int>(old_map[0].size()/2);
	vb record_bool(NN - Nprev_grid, false);
	int roller = NS*2; // Maximum number of children created
	int Nadd = 0;
	vvi tempvec(roller, vi(roller));

	for (int ii = Nprev_grid; ii < NN; ii++) {
		if (old_param[ii] >= tolsize) {
			Nadd += roller;
			record_bool[ii - Nprev_grid] = true;
		}
	}

	vvi out_vec(NN + Nadd, vi(roller));

	for (int ii = 0; ii < NN; ii++) {
		out_vec[ii] = old_map[ii];
	}

	int tempidx = 0;
	for (int ii = 0; ii < NN-Nprev_grid; ii++) {
		switch (record_bool[ii]) {
		case true: {
			tempvec = make_children_allN(old_map[Nprev_grid+ii], NS);

			for (int jj = 0; jj < roller; jj++) {
				out_vec[NN + roller*tempidx + jj] = tempvec[jj];
			}

			tempidx += 1;
			break;
		}
		default: break;
		}
	}
	int outN = NN;

	out_vec = delete_duplicates(out_vec, NN + Nadd, NS);

return make_tuple(out_vec, outN);
}

tuple<vvvi, vi> expand_all(const vvd &old_params, const vvvi &old_maps,
	const vi &Nprev_grids, double tolsize) {

	int NS = static_cast<int>(old_params.size());

	vvvi new_maps(NS);
	vi Nnew_grids(NS, 0);

	for (int ii = 0; ii < NS; ii++) {
		tie(new_maps[ii], Nnew_grids[ii]) = expand_param(old_params[ii], old_maps[ii], Nprev_grids[ii], tolsize);

		if (static_cast<int>(new_maps[ii].size()) == static_cast<int>(old_maps[ii].size())) {
			Nnew_grids[ii] = Nprev_grids[ii];
		}
	}
	
	return make_tuple(new_maps, Nnew_grids);
}

// We have level 2 by default
tuple<vvi, int> init_idxlvl(const int NS) {
	int Nold_grid = 0;
	vvi old_map(1, vi(2 * NS, 1));

	vd old_param(1, 1);
	double tolsize = 0;

	tie(old_map, Nold_grid) = expand_param(old_param, old_map, Nold_grid, tolsize);

	for (int ii = 0; ii < static_cast<int>(old_map.size()); ii++) {
		for (int jj = 0; jj < NS; jj++) {
			if (old_map[ii][jj + NS] == 1) {
				old_map[ii][jj + NS] = 2;
			}
		}
	}
	return make_tuple(old_map, Nold_grid);
}

/*
	CUDA FUNCTIONS
*/
double ProdTentsNoCUDA(const double *xx_coordinate, const int *in_idxlvl, int NS) {
	double out = 1.0;
	
	for (int ii = 0; ii < NS; ii++) {
		
		double h_l = 2.0;
		
		for (int jj = 0; jj < in_idxlvl[ii+NS]; jj++) {
			h_l /= 2.0;
		}

		if (fabs(xx_coordinate[ii] - in_idxlvl[ii] * h_l) >= h_l) {
			return 0.0;
		}
		else {
			out = out*(1.0 - fabs(xx_coordinate[ii] - in_idxlvl[ii] * h_l) / h_l);
		}
	}
	return out;
}

double EvaluateCUDA(const vd &xx_coordinate, const vd &inParams, const vvi &in_map) {
	double out = 0.0;

	int Niter = static_cast<int>(inParams.size());
	int NS = static_cast<int>(xx_coordinate.size());

	double *xx_d = new double[NS];
	double *param_d = new double[Niter];
	double *out_vec = new double[Niter];
	int *pass_idxlvl = new int[Niter*NS*2];

	for (int ii = 0; ii < Niter; ii++) {
		param_d[ii] = inParams[ii];
	}

	for (int ii = 0; ii < NS; ii++) {
		xx_d[ii] = xx_coordinate[ii];
	}

	for (int ii = 0; ii < Niter; ii++) {
		for (int jj = 0; jj < NS*2; jj++) {
			pass_idxlvl[ii*NS + jj] = in_map[ii][jj];
		}
	}

	if (Niter >= 50000) {
		ProdTentsHost(xx_d, pass_idxlvl, NS, param_d, Niter, out_vec);
		//for (int ii = 0; ii < Niter; ii++) {
		//	cout << out_vec[ii] << endl;
		//}
		//cout << "CUDA??" << endl;
	}
	else {
#pragma omp parallel for
		for (int ii = 0; ii < Niter; ii++) {
			int *temp_idxlvl = new int[NS*2];

			for (int jj = 0; jj < NS*2; jj++) {
				temp_idxlvl[jj] = in_map[ii][jj];
			}

			out_vec[ii] = ProdTentsNoCUDA(xx_d, temp_idxlvl, NS)*inParams[ii];

			delete[] temp_idxlvl;
		}
	}

	for (int ii = 0; ii < Niter; ii++) {
		out += out_vec[ii];
	}

	delete[] xx_d;
	delete[] param_d;
	delete[] pass_idxlvl;
	delete[] out_vec;

	return out;
}

tuple<vvd, vvvi, vi> UpdateAll_Init(const vvi &inMaps, const vi &inNgrids, const double tolsize) {
	int NS = static_cast<int>(inMaps[0].size()/2);

	vi outN(NS, 0);
	vvd outparams(NS);

	for (int ii = 0; ii < NS; ii++) {
		outN[ii] = inNgrids[ii];
	}

	outparams[0] = UpdateParams(inMaps, &c_init_call);
	outparams[1] = UpdateParams(inMaps, &y_init_call);
	outparams[2] = UpdateParams(inMaps, &r_init_call);
	outparams[3] = UpdateParams(inMaps, &p_init_call);
	
	vvvi outmaps{ inMaps, inMaps, inMaps, inMaps };

	cout << endl; cout << "Initialization Complete!!!" << endl; cout << endl;

	return make_tuple(outparams, outmaps, outN);
}



tuple<vvd, vvvi, vi> UpdateAll_Norm(const vvd &inParams,
	const vvvi &inMaps, const vi &inN, const int Niter1, const int Niter2,
	const double tolsize) {	
	
	int NS = static_cast<int>(inParams.size());

	vvvi outmaps(NS);
	vi outN(NS, 0);
	vvd outparams(NS);

	for (int ii = 0; ii < NS; ii++) {
		outmaps[ii] = inMaps[ii];
		outN[ii] = inN[ii];
	}

	for (int ii = 0; ii < Niter1; ii++) {
		cout << "------------------- ";
		cout << "Parameter Expansion Step : " << ii + 1 << " / " << Niter1;
		cout << " -------------------" << endl;

		for (int jj = 0; jj < Niter2; jj++) {
			clsDefModel tempObj(&EvaluateCUDA, outparams, outmaps);

			outparams[0] = UpdateParams(outmaps[0], &clsDefModel::c_norm_call, tempObj);
			outparams[1] = UpdateParams(outmaps[1], &clsDefModel::y_norm_call, tempObj);
			outparams[2] = UpdateParams(outmaps[2], &clsDefModel::r_norm_call, tempObj);
			outparams[3] = UpdateParams(outmaps[3], &clsDefModel::p_norm_call, tempObj);

			cout << "Parameter Update - " << jj + 1 << " / " << Niter2 << " Complete !!!" << endl;
		}

		tie(outmaps, outN) = expand_all(outparams, outmaps, outN, tolsize);
		
		for (int kk = 0; kk < NS; kk++) {
			cout << static_cast<int>(outmaps[kk].size()) << "\t" << outN[kk] << endl;
		}
	}
	
	return make_tuple(outparams, outmaps, outN);
}