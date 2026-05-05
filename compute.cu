#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

#define TILE 16
#define REDUCE_BLOCK 128

#define cudaCheck(call)                                                            \
	do {                                                                           \
		cudaError_t _err = (call);                                                 \
		if (_err != cudaSuccess) {                                                 \
			fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                        \
				cudaGetErrorName(_err), __FILE__, __LINE__,                        \
				cudaGetErrorString(_err));                                         \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

// Persistent scratch buffer holding the NxN pairwise acceleration matrix.
static vector3 *d_accels = NULL;

// computeAccels: fills d_accels[i*N + j] with the gravitational acceleration
// that object j exerts on object i. Uses a 16x16 tiled layout so each block
// can cooperatively load 16 row-positions, 16 column-positions, and 16
// column-masses into shared memory and reuse them across the tile.
__global__ void computeAccels(vector3 *accels, vector3 *pos, double *mass, int n)
{
	__shared__ vector3 sPosI[TILE];
	__shared__ vector3 sPosJ[TILE];
	__shared__ double  sMassJ[TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int j  = blockIdx.x * TILE + tx;
	int i  = blockIdx.y * TILE + ty;

	if (ty == 0 && j < n) {
		sPosJ[tx][0] = pos[j][0];
		sPosJ[tx][1] = pos[j][1];
		sPosJ[tx][2] = pos[j][2];
		sMassJ[tx]   = mass[j];
	}
	if (tx == 0 && i < n) {
		sPosI[ty][0] = pos[i][0];
		sPosI[ty][1] = pos[i][1];
		sPosI[ty][2] = pos[i][2];
	}
	__syncthreads();

	if (i >= n || j >= n) return;

	vector3 *out = &accels[i * n + j];
	if (i == j) {
		(*out)[0] = 0.0;
		(*out)[1] = 0.0;
		(*out)[2] = 0.0;
		return;
	}

	double dx = sPosI[ty][0] - sPosJ[tx][0];
	double dy = sPosI[ty][1] - sPosJ[tx][1];
	double dz = sPosI[ty][2] - sPosJ[tx][2];
	double mag_sq = dx*dx + dy*dy + dz*dz;
	double mag    = sqrt(mag_sq);
	double accelmag = -1.0 * GRAV_CONSTANT * sMassJ[tx] / mag_sq;

	(*out)[0] = accelmag * dx / mag;
	(*out)[1] = accelmag * dy / mag;
	(*out)[2] = accelmag * dz / mag;
}

// sumAndUpdate: one block per body. The block cooperatively reduces row i
// of the accels matrix down to a single vector3, then thread 0 applies the
// time-step integration to hVel[i] and hPos[i].
__global__ void sumAndUpdate(vector3 *accels, vector3 *vel, vector3 *pos, int n)
{
	__shared__ double sdata[3][REDUCE_BLOCK];

	int i  = blockIdx.x;
	int tid = threadIdx.x;
	if (i >= n) return;

	double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
	for (int j = tid; j < n; j += REDUCE_BLOCK) {
		vector3 *a = &accels[i * n + j];
		sum0 += (*a)[0];
		sum1 += (*a)[1];
		sum2 += (*a)[2];
	}
	sdata[0][tid] = sum0;
	sdata[1][tid] = sum1;
	sdata[2][tid] = sum2;
	__syncthreads();

	for (int s = REDUCE_BLOCK / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[0][tid] += sdata[0][tid + s];
			sdata[1][tid] += sdata[1][tid + s];
			sdata[2][tid] += sdata[2][tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		double ax = sdata[0][0];
		double ay = sdata[1][0];
		double az = sdata[2][0];
		vel[i][0] += ax * INTERVAL;
		vel[i][1] += ay * INTERVAL;
		vel[i][2] += az * INTERVAL;
		pos[i][0] += vel[i][0] * INTERVAL;
		pos[i][1] += vel[i][1] * INTERVAL;
		pos[i][2] += vel[i][2] * INTERVAL;
	}
}

extern "C" void initDeviceMemory()
{
	size_t vec_bytes  = sizeof(vector3) * NUMENTITIES;
	size_t mass_bytes = sizeof(double)  * NUMENTITIES;
	size_t mat_bytes  = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

	cudaCheck(cudaMalloc((void**)&d_hPos,   vec_bytes));
	cudaCheck(cudaMalloc((void**)&d_hVel,   vec_bytes));
	cudaCheck(cudaMalloc((void**)&d_mass,   mass_bytes));
	cudaCheck(cudaMalloc((void**)&d_accels, mat_bytes));

	cudaCheck(cudaMemcpy(d_hPos, hPos, vec_bytes,  cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_hVel, hVel, vec_bytes,  cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_mass, mass, mass_bytes, cudaMemcpyHostToDevice));
}

extern "C" void copyDeviceToHost()
{
	size_t vec_bytes = sizeof(vector3) * NUMENTITIES;
	cudaCheck(cudaMemcpy(hPos, d_hPos, vec_bytes, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(hVel, d_hVel, vec_bytes, cudaMemcpyDeviceToHost));
}

extern "C" void freeDeviceMemory()
{
	if (d_hPos)   cudaFree(d_hPos);
	if (d_hVel)   cudaFree(d_hVel);
	if (d_mass)   cudaFree(d_mass);
	if (d_accels) cudaFree(d_accels);
	d_hPos = d_hVel = NULL;
	d_mass = NULL;
	d_accels = NULL;
}

extern "C" void compute()
{
	int n = NUMENTITIES;
	int tiles = (n + TILE - 1) / TILE;

	dim3 grid2D(tiles, tiles);
	dim3 block2D(TILE, TILE);
	computeAccels<<<grid2D, block2D>>>(d_accels, d_hPos, d_mass, n);

	sumAndUpdate<<<n, REDUCE_BLOCK>>>(d_accels, d_hVel, d_hPos, n);

#ifdef DEBUG
	cudaCheck(cudaDeviceSynchronize());
#endif
}
