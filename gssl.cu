#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include "kernals.h"
#define Eu_n 2.7182818
namespace py = pybind11;


// __global__ void norm_w(int * c, int * temp,int n, int d, ){
//     // __shared__ float abc[n*n];

// }
__device__ void find_sum(float *c,float *temp, int n){
    int thread =32*32/2;
    
    while (thread > 0){
        if (32*threadIdx.x+threadIdx.y < thread){
            temp[32*threadIdx.x+threadIdx.y]+=temp[32*threadIdx.x+threadIdx.y+thread];
            temp[32*threadIdx.x+threadIdx.y+thread]=0.0;  
        }
        __syncthreads();
        thread/=2;
    }
    if (threadIdx.x ==0 && threadIdx.y==0 ) {
        atomicAdd(&c[0],2*temp[0]/((n*n)-n));
        temp[0]=0.0;
        }
}

__global__ void weight_matrix_calc(float *a,float *b, float * c,int n,int d,float* diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<n) c[i*n+i]=0.0;
    int k;
    __shared__ float temp[32*32];
    temp[32*threadIdx.x+threadIdx.y]=0.0;
    if (i<n && j <n && j>i){ 
        float s=0;
        for (k=0;k<d;k++){
            s+=pow((a[i*d+k]-b[j*d+k]),2); 
            }

        c[i*n+j]=s;
        c[i+n*j]=s;
        temp[32*threadIdx.x+threadIdx.y]=s;   

    }
    __syncthreads();

    find_sum(c,temp,n);
    // got mean at c[0]
    __syncthreads();
    

    if (i<n && j <n && i!=j){
        temp[32*threadIdx.x+threadIdx.y]=pow(c[i*n+j]-c[0],2);
        
    }
    __syncthreads();
    c[i*n+j]=temp[32*threadIdx.x+threadIdx.y];
    // __syncthreads();
    // // else temp[32*threadIdx.x+threadIdx.y]=0;

    // __syncthreads();
    // if (i ==0 && j==0 ) c[0]=0;
    // __syncthreads();
    // find_sum(c,temp,n);
    // got variance at c[0]
    // if (i<n && j <n && j>i  ) {
    //     c[i*n+j] = pow(Eu_n,-c[i*n+j]/(2*c[0]));
    //     c[i+n*j] = c[i*n+j];
    // }
    // __syncthreads();
    // // clear c[0]
    // if (i ==0 && j==0 ) c[0]=0;


    // // temp[32*threadIdx.x+threadIdx.y]=0.0;
    // if (i ==0 && j==0 ) c[0]=0;
    // __syncthreads();
    // if (i<n  ) { 
    //     int s=0;
    //     for (int p =0 ;p<n;p++){
    //         if (i!=p) {
    //             s+=c[i*n+p];
    //             }
    //     c[i*n+i]=pow(s,0.5);
    //     }
    

    // }
    // __syncthreads();
    // if (i < n && j < n && i!=j){
    //     c[i*n+j]=c[i*n+j]/(c[i*n+i]*c[j*n+j]); 
    // }
    // __syncthreads();
    // if (i<n) c[i*n+i]=0.0;
    // if (i<n) c[i*n+i]=pow(c[i*n+i],0.5);
    // __syncthreads();
    // if (i < n && j < n && i!=j){
    //     c[i*n+j]=c[i*n+j]/(c[i*n+i]*c[j*n+1]);
    // }
    // __syncthreads();
    // if (i<n) c[i*n+i]=0.0;
}


class CudaGSSL {
    public:
        float *a,*c_a;
        float *w,*c_w;
        int n=1;
        std::vector<ssize_t> shape;

        CudaGSSL(py::array_t<float> & arr){
            auto buf = arr.request();
            a = (float *)buf.ptr;
            shape = buf.shape;
            for(int i:shape){n*=i;}
            // std::cout<<n<<"#@#@#";
            w = new float [shape[0]*shape[0]];
            cudaMalloc( &c_a,n* sizeof(float));
            cudaMalloc( &c_w,shape[0]*shape[0]* sizeof(float));
        }

        py::array_t<float> get_w(){
            cudaMemcpy(w,c_w,shape[0]*shape[0]* sizeof(float),cudaMemcpyDeviceToHost);
            py::array_t<float> numpy_array({shape[0],shape[0]}, w);
            return numpy_array;    
        }

        void gen_w(){
        // void gen_w(){
            float *diag = new float [32*32];
            float *c_diag;
            cudaMalloc( &c_diag,32*32* sizeof(float));
            cudaMemcpy( c_a, a, n * sizeof(float),cudaMemcpyHostToDevice );
            // cudaMemcpy( c_a, a, n * sizeof(float),cudaMemcpyHostToDevice );
            cudaMemcpy( c_w, w, shape[0]*shape[0]* sizeof(float),cudaMemcpyHostToDevice );
            dim3 threads(32,32);
            // std::cout<<(n + threads.x - 1) / threads.x;
            dim3 grids((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
            weight_matrix_calc<<<grids,threads>>>(c_a,c_a,c_w,shape[0],shape[1],c_diag);
            cudaDeviceSynchronize();
            cudaMemcpy(diag,c_diag,32*32* sizeof(float),cudaMemcpyDeviceToHost);
            for(int x=0; x<32*32;x++)if(diag[x]!=0.0)std::cout<<diag[x]<<" ";
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
                return;
            }

            return;
        // void label_prop(){
        //     return;
        // }

        }

};

PYBIND11_MODULE(gssl,m ) { 
    py::class_<CudaGSSL>(m, "CudaGSSL") 
        .def(py::init<py::array_t<float> &>()) 
        .def("gen_w", &CudaGSSL::gen_w) 
        .def("get_w", &CudaGSSL::get_w);
        // .def_readwrite("gen_w", &test::n); //expose variable n,  rename as array_size

}
//nvcc -arch=compute_60 -code=sm_60 -O3  -shared -std=c++11 -Xcompiler -fPIC $(python3 -m pybind11 --includes) gssl.cu -o gssl$(python3-config --extension-suffix)