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
#include <chrono>
#include "AllSettings.h"

/* ------------------------------------------------------------------------------------
	Type Declarations
------------------------------------------------------------------------------------ */
vd makegrids(const vi &in_idxlvl, const int in_NS) {
	double grid_size;

	vd ret_grids(in_NS, 0.0);

	for (int ii = 0; ii < in_NS; ii++) {
	  if (in_idxlvl[ii + in_NS] == 1) {
          ret_grids[ii] = 0.5;
	  }
	  else{
          grid_size = h_1[in_idxlvl[ii + in_NS] - 1];
          ret_grids[ii] = grid_size * (double)in_idxlvl[ii];
	  }
	}

	return ret_grids;
}

vd makegrids_all(const v2i& in_idxlvl, const int in_NS, const int Niter,
	const int Nprev) {
	double grid_size;

	vd ret_grids((Niter - Nprev)*in_NS);

	for (int nn = Nprev; nn < Niter; nn++) {
		for (int ii = 0; ii < in_NS; ii++) {
			if (in_idxlvl[nn][ii + in_NS] == 1) {
				ret_grids[(nn - Nprev)*in_NS + ii] = 0.5;
			}
			else {
				grid_size = h_1[in_idxlvl[nn][ii + in_NS] - 1];
				ret_grids[(nn - Nprev) * in_NS + ii] = grid_size * (double)in_idxlvl[nn][ii];
			}
		}
	}

	return ret_grids;
}

v2i make_children(const vi &invec, int pos, const int in_NS) {
  v2i outvec(2, vi(in_NS *2, 0));

  outvec[0] = invec; outvec[1] = invec;

  int lvlidx = pos + in_NS;

  outvec[0][lvlidx] = invec[lvlidx] + 1; // Next level;
  outvec[1][lvlidx] = invec[lvlidx] + 1; // Next level;

  switch (invec[lvlidx]){
    case 1: {
      outvec[0][pos] = 0;
      outvec[1][pos] = 2;
      break;
    }
    case 2: {
      outvec[0][pos] = invec[pos] + 1;
      outvec[1][pos] = invec[pos] + 1;
      break;
    }
    default: {
      outvec[0][pos] = 2 * invec[pos] - 1; // Index 1
      outvec[1][pos] = 2 * invec[pos] + 1; // Index 2
      break;
    }
  }

  return outvec;
}

v2i make_children_allN(const vi &invec, const int in_NS) {
  v2i outvec(2 * in_NS, vi(2 * in_NS, 0));
  v2i tempvec(2, vi(2 * in_NS, 0));

  for (int ii = 0; ii < in_NS; ii++) {
    tempvec = make_children(invec, ii, in_NS);

    outvec[2 * ii] = tempvec[0];
    outvec[2 * ii + 1] = tempvec[1];
  }

  return outvec;
}

v2i delete_duplicates(const v2i &in_vec, const int untilwhen) {
  // Length of the row
  int len_row = static_cast<int>(in_vec.size());

  // Length of the columns
  int len_col = static_cast<int>(in_vec[0].size());
  int len_added = len_row - untilwhen;

  // Express levels in decimals, and start recording from fifth decimal place
  double multiplier; // We start recording from 0.00001 (Fifth place)
  double verysmall = 1e-7;

  vd decimal_lvl(len_added, 0.0);

  for (int kk = 0; kk < len_col / 2; kk++) {
    multiplier = pow(10.0, static_cast<double>(kk) - 5.0);
    for (int pp = untilwhen; pp < len_row; pp++) {
      decimal_lvl[pp - untilwhen] += static_cast<double>(in_vec[pp][kk]) * multiplier;
    }
  }

  std::vector<bool> tempbool1(len_added, false);
  bool tempbool2;
  int ll, iter_num;
  double comparevalue;

  for (int pp = untilwhen; pp < len_row; pp++) {
    vi to_search;
    comparevalue = decimal_lvl[pp - untilwhen];

    for (int kk = pp+1; kk < len_row; kk++) {
      if (std::fabs(comparevalue - decimal_lvl[kk - untilwhen]) < verysmall) {
        to_search.push_back(kk);
      }
    }

    tempbool2 = false;
    iter_num = static_cast<int>(to_search.size());

    ll = 0;
    while (!tempbool2 && ll < iter_num) {
      tempbool2 = true;

      for (int mm = 0; mm < len_col; mm++) {
        tempbool2 = tempbool2 && (in_vec[pp][mm] == in_vec[to_search[ll]][mm]);
      }

      ll += 1;
    }

    tempbool1[pp - untilwhen] = tempbool2;
  }

  int count = untilwhen;

  for (int ii = 0; ii < len_added; ii++) {
    if (!tempbool1[ii]) {
      count += 1;
    }
  }

  v2i out_vec(count, vi(len_col, 0));

  // Copy original vector -- old maps
  for (int ii = 0; ii < untilwhen; ii++) {
    out_vec[ii] = in_vec[ii];
  }

  // Copy new maps
  count = 0;
  for (int ii = untilwhen; ii < len_row; ii++) {
    if (!tempbool1[ii - untilwhen]) {
      out_vec[untilwhen + count] = in_vec[ii];
      count += 1;
    }
  }

  return out_vec;
}

v2i expand_param(const vd &old_param, const v2i &old_map, int &Nprev_grid, double tolsize, const int in_NS) {
  // Size of the previous map
  int NN = static_cast<int>(old_map.size());

  // Recording the booleans (to expand or not)
  vb record_bool(NN - Nprev_grid, false);

  int Nadd = 0, roller = in_NS * 2;	// Maximum number of children created
  v2i tempvec;	// Will be used to append rows

  for (int ii = Nprev_grid; ii < NN; ii++) {
    if (fabs(old_param[ii]) >= tolsize) {
      Nadd += roller;
      record_bool[ii - Nprev_grid] = true;
    }
  }

  v2i out_vec(NN + Nadd, vi(roller));		// The resulting vector
  for (int ii = 0; ii < NN; ii++) {
    out_vec[ii] = old_map[ii];
  }

  int tempidx = 0;
  for (int ii = 0; ii < NN - Nprev_grid; ii++) {
    switch (record_bool[ii]) {
      case true: {
        tempvec = make_children_allN(old_map[Nprev_grid + ii], in_NS);

        for (int jj = 0; jj < roller; jj++) {
          out_vec[NN + roller * tempidx +  jj] = tempvec[jj];
        }

        tempidx += 1;
        break;
      }
      default: break;
    }
  }

  out_vec = delete_duplicates(out_vec, NN);

  Nprev_grid = NN;

  return out_vec;
}

double ProdTentsNoGPU(const vd& xx_coordinate, const vi& in_idxlvl, const int in_NS) {
    double out = 1.0;
    double temp;

    for (int ii = 0; ii < in_NS; ii++) {
        temp = xx_coordinate[ii] / h_1[in_idxlvl[ii + in_NS] - 1] - static_cast<double>(in_idxlvl[ii]);
        temp = 1.0 - std::abs(temp);

        out *= std::clamp<double>(temp, 0.0, 1.0);
    }
    return out;
}

// Overloaded EvaluateNoGPU: (1) vd -> double (single point) / (2) vvd -> vd (multiple points)
double EvaluateNoGPU(const vd& xx_coordinate, const vd& inParams, const v2i& in_map,
                     const int in_NS) {
    int NV = static_cast<int>(xx_coordinate.size());

    double out = 0.0;

    int Niter = static_cast<int>(inParams.size());

    for (int jj = 0; jj < Niter; jj++) {
        out += ProdTentsNoGPU(xx_coordinate, in_map[jj], in_NS) * inParams[jj];
    }

    return out;
}

vd EvaluateNoGPU(const v2d& xx_coordinate, const vd& inParams, const v2i& in_map,
                 const int in_NS) {
    int NV = static_cast<int>(xx_coordinate.size());

    vd out(NV, 0.0);

    int Niter = static_cast<int>(inParams.size());

    for (int ii = 0; ii < NV; ii++) {
        for (int jj = 0; jj < Niter; jj++) {
            out[ii] += ProdTentsNoGPU(xx_coordinate[ii], in_map[jj], in_NS) * inParams[jj];
        }
    }

    return out;
}

vd UpdateParams(v2i &in_map, std::function<double(const vd&)> inObj, const vd &prev_params, const int in_NS) {
  /*
    Combine the grid for consumption and continuation value all together
  */
  int Niter = static_cast<int>(in_map.size());
  int Nprev = static_cast<int>(prev_params.size());

  vd outParams(Niter, 0.0), fvals(Niter - Nprev, 0.0);
  double temp_val;

  v2d eval_grids_all(Niter - Nprev, vd(in_NS, 0.0));

  for (int ii = Nprev; ii < Niter; ii++) {
    eval_grids_all[ii - Nprev] = makegrids(in_map[ii], in_NS);
  }

  // Copy paramters
  for (int row_idx = 0; row_idx < Nprev; row_idx++){
    outParams[row_idx] = prev_params[row_idx];
  }

  for (int row_idx = Nprev; row_idx < Niter; row_idx++) {
    fvals[row_idx - Nprev] = inObj(eval_grids_all[row_idx - Nprev]);
  }
  
  double tent_val, temp_tent;
  vd lookup_grids;

  for (int row_idx = Nprev; row_idx < Niter; row_idx++) {
    lookup_grids = makegrids(in_map[row_idx], in_NS);
    tent_val = 0.0;

    for (int col_idx = 0; col_idx < row_idx; col_idx++) {
      temp_tent = ProdTentsNoGPU(lookup_grids, in_map[col_idx], in_NS);
        tent_val += temp_tent * outParams[col_idx];
    }

    outParams[row_idx] = fvals[row_idx - Nprev] - tent_val;
  }
  
  return outParams;
}

// We have level 2 or 3 by default
v2i init_idxlvl(int &Nold_grid, const int in_NS, const int init_lvl) {
	v2i old_map(1, vi(2 * in_NS, 1));
    vd old_param(1, 1.0);

    double tolsize = 0.0;

    // Level 1
    for (int ii = 0; ii < in_NS; ii++){
        old_map[0][ii] = 0;
    }

    for (int jj = 1; jj < init_lvl; jj++) {
        old_map = expand_param(old_param, old_map, Nold_grid, tolsize, in_NS);
    }

    return old_map;
}