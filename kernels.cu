#include "kernels.h"
__global__ void weight_matrix_calc(double *a,double *b, double * c,int n,int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    int k;
    if (i<n && j==0) c[i*n+i]=0.0;
    if (i<n && j <n && j>i){ 
        double s=0;
        for (k=0;k<d;k++){
            s+=pow((a[i*d+k]-b[j*d+k]),2); 
        }
        c[i*n+j]=s;
        c[i+n*j]=s;
    }
}

__global__ void mat_mul(double *wi,double *y,double *res,int n, int labels){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i<n && j<labels){
        double s=0;
        for(int k=0;k<n;k++){
            s+=wi[i*n+k]*y[k*labels+j];
        }
        res[i*labels+j]=s;
    }
}

__global__ void mean(double *c,double *d ,int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = BLK*threadIdx.x+threadIdx.y;
    __shared__ double temp[BLK*BLK];
    int thread =BLK*BLK/2;
    // if(i<n) d[i]=300;
    // return;
    if (i<n &&j<n )temp[tid]=c[i*n+j];
    else temp[tid]=0.0;
    // c[0]=3;
    // return;
    __syncthreads();
    if (i<n && j<n){
    
        while (thread > 0){
            if (BLK*threadIdx.x+threadIdx.y < thread){
                temp[BLK*threadIdx.x+threadIdx.y]+=temp[BLK*threadIdx.x+threadIdx.y+thread];
                temp[BLK*threadIdx.x+threadIdx.y+thread]=0.0;  
            }
            __syncthreads();
            thread/=2;
        }
        __syncthreads();
        if (threadIdx.x ==0 && threadIdx.y==0 ) {
            // atomicAdd(&c[0],temp[0]/((n*n)-n));
            atomicAdd(&d[0],temp[0]/(n*n-n));

            __syncthreads();
        }
    }
}
    

__global__ void final_weight_matrix(double* c,double *d, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(j<n && i<n){
        c[i*n+j]=pow(Eu_n,-c[i*n+j]/(0.1*d[0])); 
    }
    
}

// only 1d thread
__global__ void find_D(double *c,double *d,int n){
    // return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n){
        double s=0;
        for(int k =0;k<n;k++){s+=c[i*n+k];}
        d[i]=pow(s,0.5);
    }
    
}
//[1.86109577 1.89569436 1.94829572 2.33572661 2.30210045 2.13042216 2.17199934 2.3522876  1.9428939 
__global__ void normalise(double *c,double *I,double *d,int n){
    // return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i<n && j<n && i!=j){
        c[i*n+j]/=(d[i]*d[j]);
        // c[i*n+j]=d[j];
        c[i*n+j]=0.9*c[i*n+j]*-1;
        
        I[i*n+j]=0.0;
        // c[i*n+j]=i*n+j;
    }
    if (i==j && i<n){
        c[i*n+j]=1.0;
        I[i*n+j]=1.0;
        // c[i*n+j]=i*n+j;
    }
}
// use 1 thread
// int min(int i,int j){
//     if (i>j)return j;
//     else return i;
// }
__global__ void non_zero(double *w, double *te,int row,int n){
    int i = threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int blocksize =BLK*BLK;
    // w[0]=9999;

    // return;
    __shared__ double temp[BLK*BLK];
    temp[i]=1024;
    // w[1]=temp[i];
    // return;
    __syncthreads();
    if(j<n){
        te[j]=w[j*n+row];
        if (j>=row && te[j]!=0){
            temp[j-row]=j;
        }
    }
    
    __syncthreads();
    // if(j==0)w[1]=99999;
    // return;
    // w[0]=9999;
    blocksize=blocksize/2;
    
    while (blocksize>0){
        if (i<blocksize){
            // temp[i]=(temp[i]>temp[i+blocksize]) ?  temp[i+blocksize] : temp[i];
            temp[i] = min(temp[i],temp[i+blocksize]);
        }
        __syncthreads();
        blocksize/=2;
    }

    __syncthreads();
    // if(j==0)w[1]=temp[0];
    if( j==0 ) te[n]=temp[0]; 
}

__global__ void swap(double *w, double *wi,double *te,int row1,int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // return;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    int row2=te[n];
    double k;
    __shared__ double temp[BLK*BLK];
    if (threadIdx.x<n) temp[threadIdx.x]=te[threadIdx.x];
    __syncthreads();
    if (i==n){
        te[row2]=te[row1];
        te[row1]=1.0;
    }
    if (i<n){
        k=w[row1*n+i];
        w[row1*n+i]=w[row2*n+i]/temp[row2];
        w[row2*n+i]=k;
    }
    else if (i<2*n){
        i-=n;
        k=wi[row1*n+i];
        wi[row1*n+i]=wi[row2*n+i]/temp[row2];
        wi[row2*n+i]=k;
    }
    


}

__global__ void  order_rows(double *w,double *wi,double *te,int row,int n){
    int k;
    // return;
    
    if (w[row*n+row]!=0){double  iii  =w[row*n+row]; for(k=0;k<n;k++){w[row*n+k]/=iii;wi[row*n+k]/=iii;}for(k=0;k<n;k++){te[k]=w[k*n+row];};return;}
    double t,s;
    for( k =row+1; k< n; k++){
        if (w[k*n+row]!=0){
            s=w[k*n+row];
            // s=1;
            for(int i=0;i<n;i++){
                t=w[row*n+i];
                w[row*n+i]=w[k*n+i]/s;
                w[k*n+i]=t;
                t=wi[row*n+i];
                wi[row*n+i]=wi[k*n+i]/s;
                wi[k*n+i]=t;
            }
            break;
        }
    }
    for(k=0;k<n;k++){te[k]=w[k*n+row];}
}

__global__ void row_opera(double *w,double *wi,double *te,int row,int n){
    // return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n && j<n && i!=row){
        w[i*n+j]-=te[i]*w[row*n+j];
        wi[i*n+j]-=te[i]*wi[row*n+j];
    }
}