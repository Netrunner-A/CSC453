#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstring>



void random_ints(int*& array, int size) {
	for (int i = 0; i < size; i++)
		array[i] = rand() % size;
}

void printArray(int* array, int size) {
	for (int i = 0; i < size - 1; i++)
		printf("%d, ", array[i]);
	printf("%d.\n", array[size - 1]);
}

int cmpfunc(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}

__global__ void rankingSort(int* in, int* out, int N) {
	int index = (blockDim.x * blockIdx.x + threadIdx.x);
	int rank = 0, same = 0;

	for (int i = 0; i < N; i++) {
		if (in[index] > in[i])
			rank++;
		if (in[index] == in[i])
			same++;
	}
	for (int i = 0; i < same; i++)
		out[rank + i] = in[index];
}


int main(){
	
	int* a, * b, * d_a, * d_b;
	int N = 50;
	int size = sizeof(int) * N;

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);

	random_ints(a, N);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	rankingSort << <1, N >> > (d_a, d_b, N);
	cudaDeviceSynchronize();

	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
	printf("After sorting, array a is: "); printArray(b, N);

	qsort(a, N, sizeof(int), cmpfunc);
	printf("Correct sort:              "); printArray(a, N);



	free(a);
	free(b);
	cudaFree(d_a);
	cudaFree(d_b);
	


	return 0;
}
