/* ------------------------------------------------------------------------------------
Define Basic NK Model

Coder : Hyok Jung Kim
Date : Jan 28th, 2019

Reference:
 - The model described below is from Chapter 1 of
   Herbst, E. P., and Schorfheide, F. (2016). 
		Bayesian Estimation of DSGE Models,
		Princeton University Press. ISBN: 9780691161082.
------------------------------------------------------------------------------------ */
#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <random>
#include <algorithm>

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

// Collection of Index and Levels
typedef vector<vector<vector<int>>> vvvi;

/* ------------------------------------------------------------------------------------
Definition of the model
------------------------------------------------------------------------------------ */

class clsDefModel {
public:
	double nu = 1 / 6, g_bar = 1.25, gamma = 1.0052;
	double bb = 0.999, tau = 2.83, phi = 17.85;

	double   psi1 = 1.8, psi2 = 0.63, z_bar = 1;
	double rho_r = 0.77, rho_g = 0.98, rho_z = 0.88;

	double sigma_r = 0.0022, sigma_g = 0.0071, sigma_z = 0.0031;

	int NM = 3, NC = 4, NS = 4;
	int maxIter = 10;

	double               p_bar = 1.0083, r_bar = gamma*p_bar / bb;
	double c_bar = pow(1 - nu, 1 / tau), y_bar = g_bar*c_bar;

	// Bounds are 40% lower and higher than steady states
	double ss_scale = 0.4;

	// Bounds
	vd c_bound{ (1 - ss_scale)*c_bar, (1 + ss_scale)*c_bar };
	vd y_bound{ (1 - ss_scale)*y_bar, (1 + ss_scale)*y_bar };
	vd r_bound{                    1, (1 + ss_scale)*r_bar };
	vd p_bound{ (1 - ss_scale)*p_bar, (1 + ss_scale)*p_bar };
	vd z_bound{ (1 - ss_scale)*z_bar, (1 + ss_scale)*z_bar };
	vd g_bound{ (1 - ss_scale)*g_bar, (1 + ss_scale)*g_bar };
	vd e_bound{ -2.5 * sigma_r, 2.5 * sigma_r };

	// Function pointers
	double(*policy_eval)(const vd&, const vd&, const vvi&);

	// Paramters of Adaptive Sparse Grids
	vd c_param, y_param, r_param, p_param;
	vvi c_idxlvl, y_idxlvl, r_idxlvl, p_idxlvl;
	vvi c_level, y_level, r_level, p_level;
	vvd param_asg;
	vvd s_bounds{ z_bound, g_bound, r_bound, e_bound };
	vvd c_bounds{ c_bound, y_bound, r_bound, p_bound };
	vvvi index_pass, level_pass;

	// Default Constructor : Do nothing
	clsDefModel() {};

	// Constructor : When the policy function is given!
	clsDefModel(double(*in_eval)(const vd&, const vd&, const vvi&),
		const vvd &in_asg, const vvvi &in_inarr) {

		// Link Function Pointers
		policy_eval = in_eval;

		// Link Adaptive Sparse Grid Parameters
		c_param = in_asg[0]; y_param = in_asg[1];
		r_param = in_asg[2]; p_param = in_asg[3];

		// Link Adaptive Sparse Grid Index and Levels
		c_idxlvl = in_inarr[0];
		y_idxlvl = in_inarr[1];
		r_idxlvl = in_inarr[2];
		p_idxlvl = in_inarr[3];
	}

	void Update(vvd &in_asg, vvvi &in_arr){
		c_param.swap(in_asg[0]);
		y_param.swap(in_asg[1]);
		r_param.swap(in_asg[2]);
		p_param.swap(in_asg[3]);

		c_idxlvl.swap(in_arr[0]);
		y_idxlvl.swap(in_arr[1]);
		r_idxlvl.swap(in_arr[2]);
		p_idxlvl.swap(in_arr[3]);
	}

	// Destructor : Just clear the vectors
	virtual ~clsDefModel() {
		vd().swap(c_param); vd().swap(y_param);
		vd().swap(r_param); vd().swap(p_param);

		vvi().swap(c_idxlvl); vvi().swap(y_idxlvl);
		vvi().swap(r_idxlvl); vvi().swap(p_idxlvl);

		vvi().swap(c_level); vvi().swap(y_idxlvl);
		vvi().swap(r_level); vvi().swap(p_idxlvl);

		vvd().swap(param_asg); vvd().swap(s_bounds);
		vvd().swap(param_asg); vvd().swap(c_bounds);

		vvvi().swap(index_pass), vvvi().swap(level_pass);
	}
	
	/*
		Abbrebiate Policy Functions: c, y, r, and p
			- Wrap function pointers for notational simplicity.
	*/
	double pol_c(const vd &inState) {
		return (*policy_eval)(inState, c_param, c_idxlvl);
	}

	double pol_y(const vd &inState) {
		return (*policy_eval)(inState, y_param, y_idxlvl);
	}

	double pol_r(const vd &inState) {
		return (*policy_eval)(inState, r_param, r_idxlvl);
	}

	double pol_p(const vd &inState) {
		return (*policy_eval)(inState, p_param, p_idxlvl);
	}

	/*
		Scale variables
			(1) fun01tolvl:   [0, 1] -> [lb, ub]
			(2) funlvlto01: [lb, ub] -> [0, 1]

			NOTE THAT THE FUNCTIONS ARE OVERLOADED!!

			- Inputs (vd, vvd) gives (vd) : we are transforming all variables
			- Inputs (vd, vd) gives (vd) : we are transforming one variables at different points
	*/
	double fun01tolvl(const double in01, const vd &inBounds) {
		double out = 0;
		
		out = (inBounds[1] - inBounds[0])*in01 + inBounds[0];
		out = max(out, inBounds[0]);
		out = min(out, inBounds[1]);

		return out;
	}

	// Change all variables at once
	vd fun01tolvl(const vd &in01, const vvd &inBounds) {
		int NV = static_cast<int>(in01.size());

		vd outVec(NV, 0);

		for (int ii = 0; ii < NV; ii++) {
			outVec[ii] = (inBounds[ii][1] - inBounds[ii][0])*in01[ii] + inBounds[ii][0];
			outVec[ii] = max(outVec[ii], inBounds[ii][0]);
			outVec[ii] = min(outVec[ii], inBounds[ii][1]);
		}

		return outVec;
	}

	// Change one variable but return a vector (several points)
	vd fun01tolvl(const vd &in01, const vd &inBounds) {
		int NN = static_cast<int>(in01.size());

		vd outVec(NN, 0);

		for (int ii = 0; ii < NN; ii++) {
			outVec[ii] = (inBounds[1] - inBounds[0])*in01[ii] + inBounds[0];
			outVec[ii] = max(outVec[ii], inBounds[0]);
			outVec[ii] = min(outVec[ii], inBounds[1]);
		}

		return outVec;
	}

	double funlvlto01(double inlvl, const vd &inBounds) {
		double out;

		out = (inlvl - inBounds[0]) / (inBounds[1] - inBounds[0]);
		out = max(out, 0.0);
		out = min(out, 1.0);

		return out;
	}

	vd funlvlto01(const vd &inlvl, const vvd &inBounds) {
		int NV = static_cast<int>(inlvl.size());

		vd outVec(NV, 0);

		for (int ii = 0; ii < NV; ii++) {
			outVec[ii] = (inlvl[ii] - inBounds[ii][0]) / (inBounds[ii][1] - inBounds[ii][0]);
			outVec[ii] = max(outVec[ii], 0.0);
			outVec[ii] = min(outVec[ii], 1.0);
		}

		return outVec;
	}

	// Change one variable but return a vector (several points)
	vd funlvlto01(const vd &inlvl, const vd &inBounds) {
		int NN = static_cast<int>(inlvl.size());

		vd outVec(NN, 0);

		for (int ii = 0; ii < NN; ii++) {
			outVec[ii] = (inlvl[ii] - inBounds[0]) / (inBounds[1] - inBounds[0]);
			outVec[ii] = max(outVec[ii], 0.0);
			outVec[ii] = min(outVec[ii], 1.0);
		}

		return outVec;
	}

	// Note: input is not 01 variable!!!
	tuple<vd, vd, vd, vd> NextState(const vd &inVars) {
		/*
			(1) Simulate R_{t}
				[Note: returns level]
		*/
		vd inVars01 = funlvlto01(inVars, s_bounds);

		double r_1 = fun01tolvl(pol_r(inVars01), r_bound);

		vd outr(NM, r_1);

		/*
			(2) Simulate e_{R,t+1}
				[Note: returns level]
		*/
		unsigned seed_e = 1;
		default_random_engine gen_e(seed_e);
		normal_distribution<double> dist_e(0, sigma_r);

		vd oute(NM, 0);

		for (int ii = 0; ii < NM; ii++) {
			oute[ii] = dist_e(gen_e);
		}

		/*
			(3) and (4) Simulate z_{t+1} and g_{t+1}
				[Note: returns level]
		*/
		unsigned seed_z = 2;
		default_random_engine gen_z(seed_z);
		normal_distribution<double> dist_z(0, sigma_z);

		unsigned seed_g = 3;
		default_random_engine gen_g(seed_g);
		normal_distribution<double> dist_g(0, sigma_g);
		
		double lng = log(g_bar);
		double lnz_1 = log(inVars[0]);
		double lng_1 = log(inVars[1]);
		
		// Initialize z_{t+1} with rho_z*ln(z_t))
		vd outz(NM, rho_z*lnz_1);

		// Initialize g_{t+1} with rho_(1-rho_{g})*ln g_t + rho_g* ln g_{t-1} ) (check equation not sure)
		vd outg(NM, (1-rho_g)*lng + rho_g*lng_1);

		for (int ii = 0; ii < NM; ii++) {
			outz[ii] = exp(outz[ii] + dist_z(gen_z));
			outg[ii] = exp(outg[ii] + dist_g(gen_g));
		}

		outz = funlvlto01(outz, z_bound);
		outg = funlvlto01(outg, g_bound);
		outr = funlvlto01(outr, r_bound);
		oute = funlvlto01(oute, e_bound);

		return make_tuple(outz, outg, outr, oute);
	}

	tuple<vvd, vd> MonteCarlo(const vd &inVars01) {
		vd temp_init(NM, 0);
		vd pass_vec(NS, 0);
		vvd outMat(NC, temp_init);

		// I think there is no need to convert from 01 to level too many times
		vd inVars = fun01tolvl(inVars01, s_bounds);

		// Performance issue!!!!!
		// Change this part by calling reference this is too inefficient
		vd z_next(NM, 0), g_next(NM, 0);
		vd r_next(NM, 0), e_next(NM, 0);

		tie(z_next, g_next, r_next, e_next) = NextState(inVars);

		for (int ii = 0; ii < NM; ii++) {
			pass_vec[0] = z_next[ii];
			pass_vec[1] = g_next[ii];
			pass_vec[2] = r_next[ii];
			pass_vec[3] = e_next[ii];

			outMat[0][ii] = pol_c(pass_vec);
			outMat[1][ii] = pol_y(pass_vec);
			outMat[2][ii] = pol_r(pass_vec);
			outMat[3][ii] = pol_p(pass_vec);
		}

		outMat[0] = fun01tolvl(outMat[0], c_bound);
		outMat[1] = fun01tolvl(outMat[1], y_bound);
		outMat[2] = fun01tolvl(outMat[2], r_bound);
		outMat[3] = fun01tolvl(outMat[3], p_bound);

		return make_tuple(outMat, z_next);
	}

	vd InitPolicy(const vd &inStates01) {
		vd inStates = fun01tolvl(inStates01, s_bounds);

		double z_t0  = inStates[0];
		double g_t0  = inStates[1];
		double r_t_1 = inStates[2];
		double er_t0 = inStates[3];

		double r_t0_next = pow(r_bar*p_bar, 1 - rho_r)*r_t_1*exp(er_t0);

		if (z_t0 < 1e-10) {
			z_t0 = 1e-10;
		}

		double p_t0_next = bb*r_t0_next / (pow(z_t0, rho_z)*gamma);

		double c_t0_next = phi*(p_t0_next - p_bar);
		c_t0_next = c_t0_next*(nu*p_t0_next - (p_t0_next - p_bar) / 2);
		c_t0_next = c_t0_next - (nu - 1);

		c_t0_next = max(c_t0_next, 0.0);
		c_t0_next = pow(c_t0_next, 1 / tau);

		double num = c_t0_next;
		double den = 1 / g_t0 - phi*(pow(p_t0_next - p_bar, 2)) / 2;

		double y_t0_next = num / den;

		vd out_array{ c_t0_next, y_t0_next, r_t0_next, p_t0_next };

		out_array = funlvlto01(out_array, c_bounds);

		return out_array;
	}

	vd NormPolicy(const vd &inStates01) {
		
		vd controls01(NC, 0);

		controls01[0] = pol_c(inStates01);
		controls01[1] = pol_y(inStates01);
		controls01[2] = pol_r(inStates01);
		controls01[3] = pol_p(inStates01);

		vd controls = fun01tolvl(controls01, c_bounds);

		double c_t0 = controls[0];
		double y_t0 = controls[1];

		vd inStates = fun01tolvl(inStates01, s_bounds);

		double g_t0 = inStates[1];
		if (g_t0 < 1e-8) {
			g_t0 = 1e-8;
		}
		double r_t_1 = inStates[2];
		double er_t0 = inStates[3];

		vd temp_init(NM, 0), z_t1(NM, 0);

		vvd NextMat(NC, temp_init);

		/*
			Fix MonteCarlo function later.
		*/

		tie(NextMat, z_t1) = MonteCarlo(inStates01);

		vd c_t1 = NextMat[0];
		vd y_t1 = NextMat[1];
		vd p_t1 = NextMat[3];

		// To compute the mean I add all up and then divide by NM
		double expectation1 = 0;
		for (int ii = 0; ii < NM; ii++) {
			expectation1 += pow(c_t1[ii]/c_t0, -tau) / (gamma*z_t1[ii] * p_t1[ii]);
		}
		expectation1 = expectation1 / NM;

		double r_t0_next = 1 / (bb*expectation1);

		r_t0_next = max(r_bound[0], r_t0_next);
		r_t0_next = min(r_bound[1], r_t0_next);

		double y_star = g_t0*pow(1 - nu, 1 / tau);

		double p_t0_next = r_bar*pow(p_bar, 1 - psi1);
		p_t0_next = pow(p_t0_next*(pow(y_t0 / y_star, psi2)), 1 - rho_r);
		p_t0_next = (p_t0_next*pow(r_t_1, rho_r) / r_t0_next)*exp(er_t0);
		p_t0_next = pow(p_t0_next, -1 / (psi1*(1 - rho_r)));

		p_t0_next = max(p_bound[0], p_t0_next);
		p_t0_next = min(p_bound[1], p_t0_next);

		double temp_num = 0;
		double expectation2 = 0;
		for (int ii = 0; ii < NM; ii++) {
			temp_num = -bb*phi*pow(c_t1[ii] / c_t0, -tau)*y_t1[ii] * p_t1[ii];
			expectation2 += temp_num*(p_t1[ii] - p_bar);
		}
		expectation2 = expectation2 / NM;

		double den = -phi*(p_t0_next - p_bar);
		den = den*(p_t0_next - (p_t0_next - p_bar) / (2 * nu));
		den = den + (1 - 1 / nu) + pow(c_t0, tau) / nu;

		double y_t0_next = expectation2 / den;

		y_t0_next = max(y_bound[0], y_t0_next);
		y_t0_next = min(y_bound[1], y_t0_next);

		double c_t0_next = 1 / g_t0 - phi*(pow(p_t0_next - p_bar, 2)) / 2;
		c_t0_next = c_t0_next*y_t0_next;
		
		c_t0_next = max(c_bound[0], c_t0_next);
		c_t0_next = min(c_bound[1], c_t0_next);

		double tol = 1e-4;
		double c_err = 0, y_err = 0, r_err = 0, p_err = 0, err_size = 1;
		int itr = 0;

		double c_t0_curr = c_t0_next, y_t0_curr = y_t0_next;
		double r_t0_curr = r_t0_next, p_t0_curr = p_t0_next;

		while (itr < maxIter && err_size > tol) {
			// Update r_t
			expectation1 = 0;
			for (int ii = 0; ii < NM; ii++) {
				expectation1 += pow(c_t1[ii] / c_t0_curr, -tau) / (gamma*z_t1[ii] * p_t1[ii]);
			}
			expectation1 = expectation1 / NM;

			r_t0_next = 1 / (bb*expectation1);

			r_t0_next = max(r_bound[0], r_t0_next);
			r_t0_next = min(r_bound[1], r_t0_next);

			// Update \pi_{t}
			p_t0_next = r_t0_next*pow(r_t_1, -rho_r) / exp(er_t0);
			p_t0_next = p_bar*pow(p_t0_next, 1 / (psi1*(1 - rho_r)));
			p_t0_next = p_t0_next*pow(r_bar*p_bar, -1 / psi1);
			p_t0_next = p_t0_next*pow(y_t0_curr / y_star, -psi2 / psi1);

			p_t0_next = max(p_bound[0], p_t0_next);
			p_t0_next = min(p_bound[1], p_t0_next);

			// Update y_{t}
			temp_num = 0;
			expectation2 = 0;
			for (int ii = 0; ii < NM; ii++) {
				temp_num = -bb*phi*pow(c_t1[ii] / c_t0_curr, -tau)*y_t1[ii] * p_t1[ii];
				expectation2 += temp_num*(p_t1[ii] - p_bar);
			}
			expectation2 = expectation2 / NM;

			den = -phi*(p_t0_next - p_bar);
			den = den*(p_t0_next - (p_t0_next - p_bar) / (2 * nu));
			den = den + (1 - 1 / nu) + pow(c_t0_curr, tau) / nu;

			y_t0_next = expectation2 / den;

			y_t0_next = max(y_bound[0], y_t0_next);
			y_t0_next = min(y_bound[1], y_t0_next);

			// Update c_{t}
			c_t0_next = 1 / g_t0 - phi*(p_t0_next - p_bar)*(p_t0_next - p_bar) / 2;
			c_t0_next = c_t0_next*y_t0_next;

			c_t0_next = max(c_bound[0], c_t0_next);
			c_t0_next = min(c_bound[1], c_t0_next);

			c_err = (c_t0_next - c_t0_curr)*(c_t0_next - c_t0_curr);
			y_err = (y_t0_next - y_t0_curr)*(y_t0_next - y_t0_curr);
			r_err = (r_t0_next - r_t0_curr)*(r_t0_next - r_t0_curr);
			p_err = (p_t0_next - p_t0_curr)*(p_t0_next - p_t0_curr);

			err_size = (c_err + y_err + r_err + p_err) / 4;

			c_t0_curr = c_t0_next; y_t0_curr = y_t0_next;
			r_t0_curr = r_t0_next; p_t0_curr = p_t0_next;

			itr += 1;
		}

		vd outvec{ c_t0_next, y_t0_next, r_t0_next, p_t0_next };

		outvec = funlvlto01(outvec, c_bounds);

		return outvec;
	}

	double c_norm_call(const vd &inStates01) {
		return NormPolicy(inStates01)[0];
	}

	double y_norm_call(const vd &inStates01) {
		return NormPolicy(inStates01)[1];
	}

	double r_norm_call(const vd &inStates01) {
		return NormPolicy(inStates01)[2];
	}

	double p_norm_call(const vd &inStates01) {
		return NormPolicy(inStates01)[3];
	}
};

double c_init_call(const vd &inStates01) {
	clsDefModel temp_c;

	return temp_c.InitPolicy(inStates01)[0];
};

double y_init_call(const vd &inStates01) {
	clsDefModel temp_y;

	return temp_y.InitPolicy(inStates01)[1];
};

double r_init_call(const vd &inStates01) {
	clsDefModel temp_r;

	return temp_r.InitPolicy(inStates01)[2];
};


double p_init_call(const vd &inStates01) {
	clsDefModel temp_p;

	return temp_p.InitPolicy(inStates01)[3];
};