// You have to implement to Odd - Even parallel sort algorithm using a unique kernel for both steps(odd step and even step).The loop should be outside the kernel.
using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include "hw1.h"


void random_ints(int*& array, int size) {
	for (int i = 0; i < size; i++)
		array[i] = rand() % size;
}

/* Used for troubleshooting */

void printArray(int* array, int size) {
	for (int i = 0; i < size-1; i++)
		printf("%d, ", array[i]);
	printf("%d.\n", array[size-1]);
}

__device__ void printArrayD(int* array, int size) {
	for (int i = 0; i < size - 1; i++)
		printf("%d, ", array[i]);
	printf("%d.\n", array[size - 1]);
}



__global__ void oddEvenSort(int *out, const int N, int c) {
	int index = (blockDim.x * blockIdx.x + threadIdx.x);

	if ((index) % 2 == c && index < N - 1 && out[index] > out[index + 1]) {
		int temp = out[index];
		out[index] = out[index + 1];
		out[index + 1] = temp;
	}
}

__global__ void oddEvenSort2(int* out, const int N) {
	int index = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ int swaps;
	__shared__ bool flag;
	bool odd;
	swaps = 1;
	odd = 1; flag = 0;

	while (true) {
		cudaThreadSynchronize;
		if (swaps == 0)
			if (flag)	// This should stop the while loop when two swap phases pass without swapping any elements (i.e: function reached solution).
				break;
			else
				flag = 1;
		swaps = 0;
		cudaThreadSynchronize;
		if ((index) % 2 == odd && index < N - 1 && out[index] > out[index + 1]) {
			int temp = out[index];
			out[index] = out[index + 1];
			out[index + 1] = temp;
			// Here the exact value of the variable is not important, as long as it gets above 0 when a swap happen to continue the loop.
			swaps++;

			// Resetting the flag whenever a swap happens.
			flag = 0;
		}

		cudaThreadSynchronize;
		odd = !odd;
	}
	cudaThreadSynchronize;
}

int cmpfunc(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}

int main() {
	int *a, *b, *d_a, *d_b;
	int N = 50;
	int size = sizeof(int) * N;
	
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);

	random_ints(a, N);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, a, size, cudaMemcpyHostToDevice);
	memcpy(b, a, size);
	printf("Array a is: "); printArray(a, N);
	
	bool odd = 1;
	for (int k = 0; k < N; k++) {
			oddEvenSort << <1, N >> > (d_a, N, odd);
			odd = !odd;
	}

	//oddEvenSort2 << <1, N >> > (d_b, N);

	cudaMemcpy(a, d_b, size, cudaMemcpyDeviceToHost);
	//cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	printf("After sorting, array a is: "); printArray(a, N);
	
	qsort(b, N, sizeof(int), cmpfunc);
	printf("Correct sort:              "); printArray(b, N);
	


	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}