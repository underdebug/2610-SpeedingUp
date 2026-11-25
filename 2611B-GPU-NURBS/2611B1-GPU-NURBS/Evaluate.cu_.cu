#include <cuda_runtime.h>

#include "Evaluate.cuh"


namespace Evaluate{
__device__ double Nabla(double* knot, int d, int i, int k) {
	int index = i + d - 1;
	return knot[index + k] - knot[index];
}


__device__ double Nrc(double* knot, int m, int r, int c, int j, int i, int d) {
    // -exec source Evaluate.cu_.py Nrc
	int I = 5;
	if(i == I){
		printf("\nm(%d), r(%d), c(%d), j(%d), i(%d), d(%d) (%d ~ %d) (%d ~ 1)", m, r, c, j, i, d, j, i - d + c - 1, r);
	}

	if (m == 0) {
		//if (((i + c - 1) == (j + d)) && (r == 1))
		if (i == I && j == i - d + c - 1){
			printf("-- 1");
		}
		if(i == I && r == 1){
			printf("-- 2");
		}

		if ((j == i - d + c - 1) && (r == 1)){
			return 1;
		}
		else{
			if (i == I){
				printf("-- 0");
			}
			return 0;
		}
	}
	else {
		return     Nabla(knot, d, i, 1)     / Nabla(knot, d, j, d - m + 1)  * Nrc(knot, m - 1, r - 1, c, j,     i, d)
			+      Nabla(knot, d, j, i - j) / Nabla(knot, d, j, d - m + 1)  * Nrc(knot, m - 1, r,     c, j,     i, d)
			+ (1 - Nabla(knot, d, j, i - j) / Nabla(knot, d, j, d - m + 1)) * Nrc(knot, m - 1, r,     c, j - 1, i, d)
			+ (  - Nabla(knot, d, i, 1)     / Nabla(knot, d, j, d - m + 1)) * Nrc(knot, m - 1, r - 1, c, j - 1, i, d);
	}
}

__global__ void GetMatrixN(double* knot, double* matrix, int span, int degree) {
	// for (int m = 0; m < degree + 1; m++) {
	// 	for (int n = 0; n < degree + 1; n++) {
	// 		matrix[m * (degree + 1) + n] = Nrc(knot, degree, m + 1, n + 1, span - (degree - 1), span - (degree - 1), degree);
	// 	}
	// }
	int m = 1, n = 0;
	matrix[m * (degree + 1) + n] = Nrc(knot, degree, m + 1, n + 1, span - (degree - 1), span - (degree - 1), degree);
}
}

void GetMatrixTest1()
{
	const double kKnots[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9};
	const int kSpan = 6;  // u = 5.1 is in [kKnots[6], kKnots[7]) = [5, 6).
	const int kDegree = 2;
	const int kKnotCount = sizeof(kKnots) / sizeof(kKnots[0]);
	const int kMatrixSize = (kDegree + 1) * (kDegree + 1);

	double* data = nullptr;
	cudaMallocManaged(&data, sizeof(kKnots) + kMatrixSize * sizeof(double));
	for (int i = 0; i < kKnotCount; ++i) {
		data[i] = kKnots[i];
	}
	double* matrix = data + kKnotCount;

	// -exec source Evaluate.cu_.py data

	Evaluate::GetMatrixN<<<1, 1>>>(data, matrix, kSpan, kDegree);
	cudaDeviceSynchronize();
	cudaFree(data);
}

void GetMatrixTest3()
{
	const double kKnots[] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9};
	const int kSpan = 8;  // u = 5.1 is in [kKnots[8], kKnots[9]) = [5, 6).

	double* data = nullptr;
	cudaMallocManaged(&data, sizeof(kKnots) + 16 * sizeof(double));
	for (int i = 0; i < 16; ++i) {
		data[i] = kKnots[i];
	}
	double* matrix = data + 16;

	Evaluate::GetMatrixN<<<1, 1>>>(data, matrix, kSpan, 3);
	cudaDeviceSynchronize();
	cudaFree(data);
}

int Main() {
	freopen("log.txt", "w", stdout);
    setvbuf(stdout, nullptr, _IONBF, 0);
    // tail -f log.txt
	printf("log");

	GetMatrixTest1();
	//GetMatrixTest3();
	return 0;
}


// GetMatrixN<<<(degree_u+cp_u),(degree_v+cp_v)>>>
