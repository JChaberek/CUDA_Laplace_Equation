
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <stdlib.h>
#include <conio.h>
#include <ctype.h>
#include <device_functions.h>
#include <string.h>
#include <inttypes.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <Python.h>

#define potential 1


//print area in the console
void print(int size,double *tab) {	
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			printf(" %.3f ", tab[y * size + x]);
		}
		printf("\n");
	}
}


//set electric potential in area to zero
__global__ void Initialization_kernel(int size,double *area_gpu) { 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	area_gpu[x*size + y] = 0;
}


//initialize Core in area
__global__ void Core_kernel(int size, int center,double* area_gpu, int* radius) { 
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	double x_dis = abs((((x + size) % size)-center));
	double y_dis = abs((y - center));
	double r = *radius;
	
	//set core potential
	if ((x_dis*x_dis)+(y_dis*y_dis) <= r*r) {
		area_gpu[x*size + y] = potential;		
	}
}


//calculate potential in the sub-area
__global__ void Calculation_kernel(double *epsilon, int center, int size, int *r, int *R , double *Buffor, double *area ) {
	int diff = *R - *r;
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int Rr = (*R) * (*R);
	double x_dis = abs((((x + size) % size) - center));
	double y_dis = abs((y - center));
	Buffor[x*size + y] = area[x*size + y];
	
	if ((x_dis*x_dis+y_dis*y_dis <= Rr)) {
		__syncthreads();
		
		for (int layer = 1; layer <= diff; layer++) {
			Buffor[x * size + y] = area[x * size + y];
			__syncthreads();
			double r_small = *r + layer - 1;
			double R_big = *r + layer;
			
			if (((x_dis*x_dis)+(y_dis*y_dis) > (r_small*r_small)) && (((x_dis*x_dis)+(y_dis*y_dis)) <= (R_big*R_big))) {
				
				//Numeric Laplace Equation
				area[x * size + y] = (Buffor[x * size + y - 1] + Buffor[x * size + y + 1] + Buffor[x * (size)+y - size] + Buffor[x * (size)+y + size]) / 4;
				__syncthreads();

				//calculate difference between actual and previous iteration
				if (area[x * size + y] >= 0.000001) {
					*epsilon = abs(area[x * size + y] - Buffor[x * size + y]);
				}
				__syncthreads();
			}
		}
		__syncthreads();
	}
}


//save results to csv file
void create_csv(int size, char* filename, double* area, int area_radius) {
	char* filename1;
	char str1[] = "Radius";
	FILE *fp, *fp1;
	filename = strcat(filename, ".csv");
	fp = fopen(filename, "w+");
	filename1 = strcat(str1, ".csv");
	fp1 = fopen(filename1, "w+");
	
	if (fp == NULL){
		printf("Unable to create a file.\n");
		exit(EXIT_FAILURE);
	}
	
	if (fp1 == NULL){
		printf("Unable to create a file.\n");
		exit(EXIT_FAILURE);
	}
	
	int newline = 0;
	for (int i = 0; i < size*size; i++) {
		
		if (((i + size) % size == 0) && (newline > 0)) {
			fprintf(fp, "\n");
			fprintf(fp, "%f,", area[i]);
		}
		
		else{ 
		fprintf(fp, "%f,", area[i]); 
		}
		newline++;
	}
	
	fprintf(fp1, "%i", area_radius);
	fclose(fp1);
	fclose(fp);
	printf("\n %s, %s files created", filename, filename1);
}


int main(){
	const int size = 32;
	const int n = 1024;
	const int center = n/2;
	bool exit_program = false;
	int choice = 10;
	int radius;
	int area_radius = 0;
	int* R_area_gpu, *r_gpu;
	double *area_gpu, *area_cpu, *Buffor ,*Buffor_cpu, *epsilon_g;
	char str[] = "Data";
	double epsilon_c = 1;
	unsigned long long int iteration = 0;
	dim3 dimblock(size, size);
	dim3 dimGrid(n / size, n / size);
	area_cpu = (double*)malloc(n * n * sizeof(double));
	Buffor_cpu = (double*)malloc(n * n * sizeof(double));
	cudaMalloc((void**)&area_gpu, n * n * sizeof(double));
	cudaMalloc((void**)&epsilon_g, sizeof(double));
	cudaMalloc((void**)&R_area_gpu, sizeof(int));
	cudaMalloc((void**)&Buffor, n * n * sizeof(double));
	cudaMalloc((void**)&r_gpu, sizeof(int));
	
	//program menu
	while (exit_program == false) {
		system("cls");
		printf("Choose operation\n");
		printf("Quit program - 0\n");
		printf("Initialize area - 1\n");
		printf("Calculate the core - 2\n");
		printf("Calculate area - 3\n");
		printf("Print actual area - 5\n");
		printf("Save into CSV file - 6\n");
		printf("Visualisation - 7\n");
		scanf_s("%i", &choice);

		if (choice == 0) {
			cudaFree(area_gpu);
			free(area_cpu);
			cudaFree(r_gpu);
			cudaFree(R_area_gpu);
			cudaFree(Buffor);
			free(Buffor_cpu);
			cudaFree(epsilon_g);
			exit_program = true;
		}
	

		if (choice == 1) {
			cudaMemcpy(Buffor, Buffor_cpu, n * n * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(area_gpu, area_cpu, n * n * sizeof(double), cudaMemcpyHostToDevice);
			Initialization_kernel << <dimblock, dimGrid >> > (n,area_gpu);
			Initialization_kernel << <dimblock, dimGrid >> > (n,Buffor);
			cudaMemcpy(area_cpu, area_gpu, n * n * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Buffor_cpu, Buffor, n * n * sizeof(double), cudaMemcpyDeviceToHost);
		}


		if (choice == 3) {
			iteration = 0;
			epsilon_c = 1;
			printf("\nArea radius: ");
			scanf_s("%i", &area_radius);
			cudaMemcpy(area_gpu, area_cpu,n * n * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(R_area_gpu, &area_radius, sizeof(int), cudaMemcpyHostToDevice);
			while (epsilon_c > 0.000000015) {
				cudaMemcpy(epsilon_g, &epsilon_c, sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(Buffor, Buffor_cpu, n * n * sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(area_gpu, area_cpu, n * n * sizeof(double), cudaMemcpyHostToDevice);
				Calculation_kernel << < dimblock, dimGrid >> > (epsilon_g, center, n, r_gpu, R_area_gpu, Buffor, area_gpu);
				cudaMemcpy(Buffor_cpu, Buffor, n * n * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(area_cpu, area_gpu, n * n * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(&epsilon_c, epsilon_g, sizeof(double), cudaMemcpyDeviceToHost);
				iteration = iteration + 1;
				if (iteration > n*2) {
					epsilon_c = 0;
				}
			}
			
			printf("iterations %llu\n", iteration);
			printf("calculations done..\n");
			char mychar;
			scanf("%c", &mychar);
			getchar();
			iteration = 0;
			epsilon_c = 1;
		}


		if (choice == 5) {
			printf("Electrical potential for this area \n");
			print(n, area_cpu);
			printf("calculations done..\n");
			char mychar;
			scanf("%c", &mychar);
			getchar();
		}


		if (choice == 2) {
			printf("Core radius: ");
			scanf_s("%i", &radius);
			cudaMemcpy(r_gpu, &radius, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(area_gpu, area_cpu, n * n * sizeof(double), cudaMemcpyHostToDevice);
			Core_kernel << <dimblock,dimGrid >> > (n,center,area_gpu, r_gpu);
			cudaMemcpy(area_cpu, area_gpu, n * n * sizeof(double), cudaMemcpyDeviceToHost);
		}


		if (choice == 6) {
			char str[] = "Data";
			create_csv(n, str, area_cpu, area_radius);
			char mychar;
			scanf("%c", &mychar);
			getchar();
		}

		//Python visualisation
		if (choice == 7) {
			char path[] = "simulation.py";
			FILE* fp;
			int argc = 1;
			wchar_t* argv[1];
			argv[0] = L"simulation.py";
			Py_Initialize();
			Py_SetProgramName(argv[0]);
			PySys_SetArgv(argc, argv);
			fp = _Py_fopen(path, "r");
			PyRun_SimpleFile(fp, path);
			Py_Finalize();
			_getch();
		}
	}
    
	return 0;
}


