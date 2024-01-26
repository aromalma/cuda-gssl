













__global__ void inverse(float *a,float *b,int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int k=0,p;
    __shared__ float temp1[32*32];
    __shared__ float temp2[32*32];
    if (i<n && j<n){
        if(i==j)b[i*n+j]=1.0;
        else b[i*n+j]=0.0;
    }

    for (; k<n; k++){
        
        // if (k==1)break;
        if (i==k && j<n){
            p=k;
            while (p<n){
                if (a[p*n+k]!=0){break; }
                p++;
            }
            temp2[tx*32+ty]=b[p*n+j]/a[p*n+k];
            temp1[tx*32+ty]=a[p*n+j]/a[p*n+k];
            
            __syncthreads();
            a[p*n+j]=a[k*n+j];
            b[p*n+j]=b[k*n+j];

            a[k*n+j]=temp1[tx*32+ty];
            b[k*n+j]=temp2[tx*32+ty];

            
        } 
        __syncthreads();
        if (i!=k && j<n && i<n){
            if (a[i*n+k]!=0){

                temp2[tx*32+ty]=b[i*n+j]-b[k*n+j]*a[i*n+k];
                temp1[tx*32+ty]=a[i*n+j]-a[k*n+j]*a[i*n+k];
                
                __syncthreads();
                a[i*n+j]=temp1[tx*32+ty];
                b[i*n+j]=temp2[tx*32+ty];

            }
            
        }    
        __syncthreads();       
    }
    
}
