#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
namespace py = pybind11;

__global__ void weight_matrix_calc(float *a,float *b, float * c,int n,int d) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // int k = blockIdx.z * blockDim.z + threadIdx.z; 
    int k;
    // weight matrix is symmetric only cal for upper traingle
    if (i<n && j <n && j>i  ) { 
        int s=0;
        for (k=0;k<d;k++){
            s+=pow((a[i*d+k]-b[j*d+k]),2); 
            }
        c[i*n+j]=s;
        c[i+n*j]=s;
    }
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)c[i*n+i]=0.0;
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

            cudaMemcpy( c_a, a, n * sizeof(float),cudaMemcpyHostToDevice );
            cudaMemcpy( c_w, w, shape[0]*shape[0]* sizeof(float),cudaMemcpyHostToDevice );
            dim3 threads(32,32);
            dim3 grids((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);;
            weight_matrix_calc<<<grids,threads>>>(c_a,c_a,c_w,shape[0],shape[1]);
            cudaDeviceSynchronize();
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
                return;
            }

            return;
        }



};

PYBIND11_MODULE(gssl,m ) { // cpp_sort is module name , m is interface for binding  
    py::class_<CudaGSSL>(m, "CudaGSSL") // exposes the class to python as cpp_ 
        .def(py::init<py::array_t<float> &>()) //  expose the class constructor function,  array as input
        .def("gen_w", &CudaGSSL::gen_w) // expose the heapsort function from cpp_ class
        .def("get_w", &CudaGSSL::get_w);
        // .def_readwrite("gen_w", &test::n); //expose variable n,  rename as array_size

}
//nvcc -arch=compute_60 -code=sm_60 -O3  -shared -std=c++11 -Xcompiler -fPIC $(python3 -m pybind11 --includes) gssl.cu -o gssl$(python3-config --extension-suffix)