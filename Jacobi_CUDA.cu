#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string>
#include <iostream>
#include <time.h>
#include <sys/time.h>

int MAX_ITER =  1000;
int TEST_TIME = 1;
__constant__ double THRESHOLD = 1e-9;
double h_THRESHOLD = 1e-9;

using namespace std;

int isConvergeHost(double *cur_x, double *pre_x, int row){
        double diff = 0.0;
        for (int i = 0 ; i < row;i++){
            diff = diff + pow(fabs(cur_x[i]-pre_x[i]),2);
        }
        if (diff < h_THRESHOLD){
            return 1;
        }  
        return 0;  
}

__device__ 
int isConverge(double *cur_x, double *pre_x, int row){
        double diff = 0.0;
        for (int i = 0 ; i < row;i++){
            diff = diff + pow(fabs(cur_x[i]-pre_x[i]),2);
        }
        if (diff < THRESHOLD){
            return 1;
        }  
        return 0;  
}


// Host version of the Jacobi method
void doSequentailJacobi(double *h_cur_x, double *h_pre_x, double *h_A, double *h_b, int row, int col){
    for (int i = 0 ; i < row; i++){
        double sigma = 0.0;
        for (int j = 0 ; j < col; j++){
             if (i!=j){
                 sigma += h_A[i*col+j]*h_pre_x[j];
            }
        }
        h_cur_x[i] = (h_b[i] - sigma) / h_A[i*col+i];
    }
}

// Device version of the Jacobi method
__global__ 
void parallelJacob(double *cur_x, double *pre_x, double *A, double *b, int row, int col, int *isCon){

    double sigma = 0.0;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int preCom = idx*col;
    for (int j=0; j<col; j++){
        if (idx != j)
            sigma += A[preCom+ j] * pre_x[j];
    }
    cur_x[idx] = (b[idx] - sigma) / A[preCom + idx];
    //Synchronize Threads to determine whether we converge here
    //__syncthreads();
    //*isCon = isConverge(cur_x, pre_x, row);

}


void checkCudaSucess(cudaError_t cudaStatus, string str){
    if (cudaStatus != cudaSuccess) {
        cout << cudaStatus << endl;
        fprintf(stderr, "%s failed!\n", str.c_str());
    }
}

void normalJacobi(double *h_cur_x, double *h_pre_x, double *h_A, double *h_b, int row, int col){
    
    for (int k=0; k<MAX_ITER; k++) {
        if (k%2)
            doSequentailJacobi (h_cur_x, h_pre_x, h_A, h_b, row, col);
        else
            doSequentailJacobi (h_pre_x, h_cur_x, h_A, h_b, row, col);
/*        
        if (isConvergeHost(h_cur_x, h_pre_x, row)) {
            cout << k << endl;
            break;
        }           
*/
    } 
}

void cudaJacobi(int nBlocks, int blockSize, double *d_cur_x, double *d_pre_x, double *d_A, double *d_b, int row, int col, int *d_isConverge, int h_isConverge){
    //Parallel
    for (int k=0; k<MAX_ITER; k++){
        if (k%2)
            parallelJacob <<< nBlocks, blockSize >>> (d_cur_x, d_pre_x, d_A, d_b, row, col, d_isConverge);
        else
            parallelJacob <<< nBlocks, blockSize >>> (d_pre_x, d_cur_x, d_A, d_b, row, col, d_isConverge);
/*
        checkCudaSucess(cudaMemcpy(&h_isConverge, d_isConverge, sizeof(int), cudaMemcpyDeviceToHost),"cudaMemcpy-h_isConverge"); 

        if (h_isConverge) {
            cout << k << endl;
            break;
        } 
*/  
    }    
}

//Check whether there is any GPU available and ready to use
bool InitCUDA(){
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

double time_diff(struct timeval x , struct timeval y){
    double x_ms , y_ms , diff;
     
    x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
    y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
     
    diff = (double)y_ms - (double)x_ms;
     
    return diff;
}

int main(int argc, char *argv[]){
    
    if(!InitCUDA()) {
        return -1;
    }

    //Time counter initialization
    struct timeval h_start, h_stop, d_start, d_stop;
    double diff;

    //Device variables
    double *d_cur_x, *d_pre_x;
    double *d_A, *d_b;
    int *d_isConverge = 0;

    //Host variables
    double *h_cur_x, *h_pre_x;
    double *h_A, *h_b; 
    int h_isConverge = 0;

    // Read Matrix From file
    FILE* file;
    file = fopen("matrix.txt", "r");
    if (file == NULL){
        fprintf(stderr, "File does not exist!\n");
        return -1;
    }

    //Get Matrix From File
    char *line;
    int N;
    size_t len = 0;
    if (getline(&line, &len, file) != -1){
        N = atoi(line);
    } else {
        return -1;
    }
    int row  = N;
    int col = N;

    h_A = (double*) malloc(sizeof(double)*row*col);
    h_b = (double*) malloc(sizeof(double)*col);

    int i=0;
    while ((getline(&line, &len, file)) != -1) {
        if (i<N*N)
            h_A[i] = atof(line);
        else
            h_b[i-N*N] = atof(line);
        i++;
    }

    double *h_x = (double*) malloc(sizeof(double)*row);

    //////////////////////////   
    // ans = {1,0,2}
    //double h_A[row*col] = {3.0,2.0,3.0,2.0,5.0,-7.0,1.0,2.0,-2.0};
    //double h_b[row] = {9.0,-12.0,-3.0};

    // ans = {1,-3,2}
    //double h_A[row*col] = {4.0,2.0,3.0,3.0,-5.0,2.0,-2.0,3.0,8.0};
    //double h_b[row] = {8.0,-14.0,27.0};    
    /////////////////////////


    // Malloc Memory in Device
    checkCudaSucess(cudaMalloc((void**) &d_cur_x, sizeof(double)*row), "cudaMalloc-d_cur_x");
    checkCudaSucess(cudaMalloc((void**) &d_pre_x, sizeof(double)*row), "cudaMalloc-d_pre_x");
    checkCudaSucess(cudaMalloc((void**) &d_A, sizeof(double)*row*col), "cudaMalloc-d_A");
    checkCudaSucess(cudaMalloc((void**) &d_b, sizeof(double)*row), "cudaMalloc-d_b");
    checkCudaSucess(cudaMalloc((void**) &d_isConverge, sizeof(int)), "cudaMalloc-d_isConverge");

    // Malloc Memory in Host
    h_cur_x = (double*) malloc(sizeof(double)*row);
    h_pre_x = (double*) malloc(sizeof(double)*row);

    // Initialize our Guess X = [0] in Device;
    checkCudaSucess(cudaMemset(d_cur_x, 0, sizeof(double)*row), "cudaMemset-d_cur_x");
    checkCudaSucess(cudaMemset(d_pre_x, 0, sizeof(double)*row), "cudaMemset-d_pre_x");

    // Initialize our Guess X = [0] in Host;
    memset(h_cur_x, 0, sizeof(double)*row);
    memset(h_pre_x, 0, sizeof(double)*row);

    // Copy memory data from host to device 
    checkCudaSucess(cudaMemcpy(d_A, h_A, sizeof(double)*row*col, cudaMemcpyHostToDevice), "cudaMemcpy-d_A");
    checkCudaSucess(cudaMemcpy(d_b, h_b, sizeof(double)*row, cudaMemcpyHostToDevice), "cudaMemcpy-d_b");
    checkCudaSucess(cudaMemcpy(d_isConverge, &h_isConverge, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy-d_isConverge");

    //Count Execution Time For Host
    diff = 0.0;
    for (int test = 0 ; test < TEST_TIME; test++){    
        gettimeofday(&h_start, NULL);
        normalJacobi(h_cur_x, h_pre_x, h_A, h_b, row, col);
        gettimeofday(&h_stop, NULL);
        diff += time_diff(h_start , h_stop);
        //printf("Host computation tooks: %.0lf us\n" , time_diff(h_start , h_stop) ); 
    }
    printf("Host computation tooks: %.0lf us\n" , diff/TEST_TIME); 


    // For CUDA use
    int blockSize = row;
    int nBlocks = 1;
    
    //Count Execution Time For Device
    diff = 0.0;
   for (int test = 0 ; test < TEST_TIME; test++){
        gettimeofday(&d_start, NULL);
        cudaJacobi(blockSize,nBlocks,d_cur_x, d_pre_x, d_A, d_b, row, col,d_isConverge,h_isConverge);
        gettimeofday(&d_stop, NULL);
        diff += time_diff(d_start , d_stop);
        //printf("Device computation tooks: %.0lf us\n" , time_diff(d_start , d_stop)); 
    }
    printf("Device computation tooks: %.0lf us\n" , diff/TEST_TIME); 

    // Data <- device
    cudaMemcpy(h_x, d_cur_x, sizeof(double)*col, cudaMemcpyDeviceToHost);
 /*  
    //Print the result for comparison
    for (int i = 0 ; i < col; i++){
        cout << h_cur_x[i] << endl;
        cout << h_x[i] << endl;
    }
*/
    //Cuda Free
    cudaFree(d_cur_x); 
    cudaFree(d_pre_x); 
    cudaFree(d_A); 
    cudaFree(d_b);

    //Normal Free
    free(h_x);
    free(h_cur_x);
    free(h_pre_x);

    return 0;
}

/*
REF:
https://github.com/MMichel/CudaJacobi/blob/master/jacobi.cu
*/

