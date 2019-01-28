/* ------------------------------------------------------------------------------------
CUDA code for calculating the policy function

Coder: Hyok Jung Kim

This code is CUDA version of ProdTents() in AdaptiveSparseGrid.h
------------------------------------------------------------------------------------ */
#include "ProdTents.cuh"
#include "stdio.h"

using namespace std;

__global__ void ProdTentsKernel(const double *xx_coordinate,
	const int *in_idxlvl, const int NS, const int Niter,
	double *out, const double *inParams) {

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	
	// Evaluate rows that are relevant
	if (row < Niter) {
		int idxrow = row*NS * 2;
		double h_l = 2.0;
		double tempval = 1.0;

		out[row] = 1;

		#pragma unroll
		for (int ii = 0; ii < NS; ii++) {
			h_l = 2.0;

			#pragma unroll
			for (int jj = 0; jj < in_idxlvl[idxrow + ii + NS]; jj++) { h_l /= 2.0; }

			tempval = fmin(1.0 - (xx_coordinate[ii] - h_l*in_idxlvl[idxrow + ii]) / h_l, 1.0 + (xx_coordinate[ii] - h_l*in_idxlvl[idxrow + ii]) / h_l);
			tempval = fmax(tempval, 0.0);
			out[row] *= tempval;
		}
		out[row] *= inParams[row];
	}
}

__declspec(dllexport) void ProdTentsHost(const double *xx_coordinate, const int *in_idxlvl,
	const int NS, const double *inParams, const int Niter, double *out_vec){

	size_t xx_memsize = NS * sizeof(double);
	size_t param_memsize = Niter * sizeof(double);
	size_t matsize = NS * Niter * 2 * sizeof(int);
	// cudaError_t error;
	/* For future use to check errors
	if (error != cudaSuccess)
	{
		printf("Error : %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
	*/

	/*
		(1) Pass grid point
	*/
	double *xx_pass;
	cudaMalloc(&xx_pass, xx_memsize);
	cudaMemcpy(xx_pass, xx_coordinate, xx_memsize, cudaMemcpyHostToDevice);

	/*
		(2) Pass indexes and levels as matrices
	*/
	int *idxlvl_pass;
	cudaMalloc(&idxlvl_pass, matsize);
	cudaMemcpy(idxlvl_pass, in_idxlvl, matsize, cudaMemcpyHostToDevice);

	/*
		(3) Pass parameters
	*/
	double *param_pass;
	cudaMalloc(&param_pass, param_memsize);
	cudaMemcpy(param_pass, inParams, param_memsize, cudaMemcpyHostToDevice);
	
	/*
		(4) Out memory for the DEVICE
	*/
	double *out_pass;
	cudaMalloc(&out_pass, param_memsize);

	/*
		Set Dimensions
			- I assume that we have 1024 cores per each grid
	*/
	int ngridx = static_cast<int>(ceil((double)Niter / 1024.0));

	/*
		Evaluate
	*/
	ProdTentsKernel<<<ngridx, 1024>>>(xx_pass, idxlvl_pass, NS, Niter, out_pass, param_pass);
	
	/*
		Copy back to the Host memory
	*/
	cudaMemcpy(out_vec, out_pass, param_memsize, cudaMemcpyDeviceToHost);

	/*
		Free Device memory
	*/
	cudaFree(xx_pass); cudaFree(idxlvl_pass);
	cudaFree(param_pass); cudaFree(out_pass);
}