
#ifndef KERNELS_H
#define KERNELS_H
#define Eu_n 2.7182818
#define BLK 32
#include <iostream>
#include <math.h>
__global__ void weight_matrix_calc(double *a,double *b, double * c,int n,int d);
__global__ void mat_mul(double *wi,double *y,double *res,int n, int labels);
__global__ void inverse(double *a,double *b,int n);
__global__ void mean(double *c,double *d ,int n);
__global__ void final_weight_matrix(double* c,double *d, int n);
__global__ void find_D(double *c,double *d,int n);
__global__ void normalise(double *c,double *I,double *d,int n);
__global__ void non_zero(double *w, double *te,int row,int n);
__global__ void swap(double *w, double *wi,double *te,int row1,int n);
__global__ void  order_rows(double *w,double *wi,double *te,int row,int n);
__global__ void row_opera(double *w,double *wi,double *te,int row,int n);
#endif