#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <math.h>
#include "kernels.h"
namespace py = pybind11;

class CudaGSSL {
    public:
        double *a,*c_a;
        double *w,*c_w,*c_wi;
        int n=1;
        std::vector<ssize_t> shape;
        double *c_te,*te;

        CudaGSSL(py::array_t<double> & arr){
            auto buf = arr.request();
            a = (double *)buf.ptr;
            shape = buf.shape;
            for(int i:shape){n*=i;}
            // std::cout<<n<<"#@#@#";
            w = new double [shape[0]*shape[0]];
            te= new double [shape[0]+1];
            cudaMalloc( &c_a,n* sizeof(double));
            cudaMalloc( &c_w,shape[0]*shape[0]* sizeof(double));
            cudaMalloc( &c_wi,shape[0]*shape[0]* sizeof(double));
            for(int k=0;k<shape[0];k++){te[k]=0.0;}
            cudaMalloc(&c_te,(shape[0]+1)*sizeof(double));
            cudaMemcpy( c_te, te, (shape[0]+1) * sizeof(double),cudaMemcpyHostToDevice );
        }
        ~CudaGSSL(){
            delete [] w;
            delete [] te;
            cudaFree(c_a);
            cudaFree(c_wi);
            cudaFree(c_w);
            cudaFree(c_te);
            
        }
        py::array_t<double> get_wi(){
            double *wi=new double [shape[0]*shape[0]];
            cudaMemcpy(wi,c_wi,shape[0]*shape[0]* sizeof(double),cudaMemcpyDeviceToHost);

            py::array_t<double> numpy_array({shape[0],shape[0]}, wi);
            return numpy_array;    
        }

        py::array_t<double> get_w(){
            cudaMemcpy(w,c_w,shape[0]*shape[0]* sizeof(double),cudaMemcpyDeviceToHost);
            py::array_t<double> numpy_array({shape[0],shape[0]}, w);
            return numpy_array;    
        }

        void gen_w(){


            cudaMemcpy( c_a, a, n * sizeof(double),cudaMemcpyHostToDevice );

            dim3 block_dim(BLK,BLK) ;
            int threadsPerBlock = 256;
            int blocksPerGrid = (shape[0] + threadsPerBlock - 1) / threadsPerBlock;
            dim3 grids_dim((n*2 + block_dim.x - 1) / block_dim.x, (n*2 + block_dim.y - 1) / block_dim.y);
            weight_matrix_calc<<<grids_dim,block_dim>>>(c_a,c_a,c_w,shape[0],shape[1] );
            cudaDeviceSynchronize();
            // GET MEAN OF ELEMENTS
            mean<<<grids_dim,block_dim>>>(c_w, c_te,shape[0]);
            cudaDeviceSynchronize();
            //SIGMA^2 TAKEN AS 0.05*MEAN
            final_weight_matrix<<<grids_dim,block_dim>>>(c_w, c_te,shape[0]);
            cudaDeviceSynchronize();
            // FIND D^0.5 FOR NORMALIZE
            find_D<<<blocksPerGrid,threadsPerBlock>>>(c_w,c_te,shape[0]);
            cudaDeviceSynchronize();
            // D W D
            normalise<<<grids_dim,block_dim>>>(c_w,c_wi,c_te,shape[0]);
            cudaDeviceSynchronize();
   
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
                return;
            }

            return;
        }

        py::array_t<double> label_prop(py::array_t<double>  arr){
        // void label_prop(py::array_t<double> & arr){
            auto buf = arr.request();
            double *y,*c_y; //*c_wi,
            
            y = (double *)buf.ptr;
            
            // t_c= new double [shape[0]];
            shape = buf.shape;
            dim3 block_dim(BLK,BLK);
            dim3 grids_dim((2*n + block_dim.x - 1) / block_dim.x, (2*n + block_dim.y - 1) / block_dim.y);

            cudaMalloc(&c_y, shape[0]*shape[1]*sizeof(double));
            cudaMemcpy(c_y,y , shape[0]*shape[1]*sizeof(double),cudaMemcpyHostToDevice );

            // for find inverse 
            for(int k=0;k<shape[0];k++){
                // break;
                // cudaDeviceSynchronize();
                // non_zero<<<(shape[0]+1023)/1024,1024>>>(c_w,c_te,k,shape[0]);
                // cudaDeviceSynchronize();
                // if (k==0)break;
                // swap<<<(2*shape[0]+(BLK*BLK)-1)/(BLK*BLK),BLK*BLK>>>(c_w, c_wi,c_te,k,shape[0]);

                // cudaDeviceSynchronize();
                //TODO: can be more parallelizable
                order_rows<<<1,1>>>(c_w,c_wi,c_te,k,shape[0]);
                cudaDeviceSynchronize();
   
                row_opera<<<grids_dim,block_dim>>>(c_w,c_wi,c_te,k,shape[0]);
                cudaDeviceSynchronize();

            }

            double *res;
            cudaMalloc(&res, shape[0]*shape[1]*sizeof(double));
            mat_mul<<<grids_dim,block_dim>>>(c_wi,c_y,res,shape[0],shape[1]);

            cudaMemcpy(y,res,shape[0]*shape[1]* sizeof(double),cudaMemcpyDeviceToHost);
            py::array_t<double> numpy_array({shape[0],shape[1]}, y);
            return numpy_array;    
        }
};

PYBIND11_MODULE(gssl,m ) { 
    py::class_<CudaGSSL>(m, "CudaGSSL") 
        .def(py::init<py::array_t<double> &>()) 
        .def("gen_w", &CudaGSSL::gen_w) 
        .def("get_w", &CudaGSSL::get_w)
        .def("get_wi", &CudaGSSL::get_wi)
        .def("label_prop",&CudaGSSL::label_prop);
        

}
//nvcc -arch=compute_60 -code=sm_60 -O3  -shared -std=c++11 -Xcompiler -fPIC $(python3 -m pybind11 --includes) gssl.cu -o gssl$(python3-config --extension-suffix)