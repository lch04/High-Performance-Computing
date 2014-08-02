
/*American Option Pricing using parallel projected sor method*/

#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas.h>


//const int N=4999;
const int threadsPerBlock=256;
const int blockSize=100;
//const int blockPerGrid = min(blockSize, (N+threadsPerBlock-1)/threadsPerBlock);

void psor(const double * Diag, const double * SuperDiag, const double * SupDiag, double * b, double * x, const double * payoff, int n, double om, int blockPerGrid);

double norm(double *a, double *b, int n);

int main(){
      int i=0,totalNums;
	int S_N[100];
	double om[100];
	//double atod ( const char * str );
	char line[100]; /* declare a char array */
	FILE *file; /* declare a FILE pointer */
	file = fopen("N.txt", "r"); /* open a text file for reading */
	while(fgets(line, sizeof line, file)!=NULL)
	{ /* keep looping until NULL pointer... */
	    S_N[i]=atoi(line); /* convert string to double float*/
	    i++;
	}
	totalNums = i;
	i = 0;
	file = fopen("om.txt", "r");
	while(fgets(line, sizeof line, file)!=NULL)
	{ /* keep looping until NULL pointer... */
	    om[i]=atof(line); /* convert string to double float*/
	    i++;
	}
	i = 0;
	fclose(file);
	
    int j = 32;
    double Smin =0.0;
    double Smax =100.0;
    double S =50.0;
    double rate = 0.1;
    double strike =50.0;
    double T = 5.0/12.0;
    double sigma = 0.4;
    int M =100;
    //int S_N =N+1; // pick the stock discrete step as the thread N fucntion
    //int i; /* iterator */
    double * M1Diag;   /* diagnoal elements of A */
    double * M1SuperDiag; /* superdiagonal elements of A */
    double * M1SupDiag; /* supdiagonal elements of A */
    double * b; /* vector in Ax=b */	
    double dt = T/M;

   for (j=0; j<totalNums ; j++){
   //    printf("N is %d, om is %0.16f\n",S_N[j],om[j]);
   // j = 0;
    double ds = (Smax - Smin) / S_N[j];
    int n = (S_N[j] - 1); 
    int blockPerGrid = min(blockSize, (S_N[j]+threadsPerBlock-2)/threadsPerBlock);

	M1Diag = (double *)malloc(n*sizeof(double));
	M1SuperDiag = (double *)malloc(n*sizeof(double));
	M1SupDiag = (double *)malloc(n*sizeof(double));
	b = (double *)malloc(n*sizeof(double));
	
	
    M1Diag[n-1] =  1.0 + dt * (sigma * sigma * n * n + rate );
	
    double sup = 0.5 * dt * (sigma * sigma * 1 - rate * 1);
    double super =  0.5 * dt * (sigma * sigma * n * n + rate * n) ;
	
    for(i=1; i<n; ++i){
        M1Diag[i-1] = 1.0 + dt * (sigma * sigma * (double) (i * i) + rate ) ;
        M1SuperDiag[i-1] = -0.5 * dt * (sigma * sigma * (double)(i * i) + rate * (double)i) ;
        M1SupDiag[i-1] = -0.5 * dt * (sigma * sigma * (i +1 )* (i+1) - rate * (i+1)) ;
    }
	
	M1SuperDiag[n-1] = 0.0;
	M1SupDiag[n-1] = 0.0;
	
    double * payoff;
	payoff = (double *)malloc(n*sizeof(double));
	
    for(i=1; i< S_N[j]; ++i ){
        payoff[i-1] = max(strike - ds * i,0.0);
    }
    double v1 = max(strike - ds * 0.0, 0.0);
    double v2 = max(strike - ds * S_N[j], 0.0);
	
    double *x;
	x = (double *)malloc(n*sizeof(double));
    memcpy( x, payoff, n*sizeof(double));
	
    clock_t start = clock();
    for(i=1; i<=M; ++i){
        memcpy( b, x, n*sizeof(double));
        b[0] += sup * v1;
        b[n-1] += super * v2;
		
        psor( M1Diag, M1SuperDiag, M1SupDiag, b,  x, payoff, n, om[j],blockPerGrid);
    }
    clock_t end = clock();
    double cpu_time = (double)( end - start ) / CLOCKS_PER_SEC;
	
    int idx = S/ds - 1;
    printf("American Option Price is %0.16f ", x[idx]);
    printf("Computing time is %0.16f \n", cpu_time);
    //printf("Blocksize is %d \n", blockPerGrid);
	
    free(x);
    free(b);
    free(M1Diag);   
    free(M1SuperDiag); 
    free(M1SupDiag); 
    free(payoff); 
	}
    return 0;
}



/*************************************************************************/
/*                       FUNCTION NORM                                   */
/*************************************************************************/
/*                 Computes the 2-norm ||a-b||                           */
/*************************************************************************/
double norm(double *a, double *b, int n){
	int j;
	double z;
	z = 0.0;
	
	for (j = 0; j < n; ++j){
		z +=(a[j]-b[j])*(a[j]-b[j]);
	}
	z = sqrt(z);
	return z;
}

/*************************************************************************/
/*                   FUNCTION Parallel SOR (Red Black SOR)               */
/*************************************************************************/
/*                                                                       */
/*************************************************************************/

__global__ void psorHelper(double * dev_x, const double * dev_y, double * dev_partial_sum, const double * dev_Diag, const double * dev_SuperDiag, const double * dev_SupDiag, const double * dev_b, const double * dev_payoff, double dev_om, int N)
{
    int tid, prev, next;
	double tmp = 0;
	
	__shared__ double cache[threadsPerBlock];
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;
	
	
    prev = tid -1; 
    next = tid +1;
	if(tid>0 && tid < N-1 && tid%2==0){
		tmp = (dev_b[tid] - (dev_x[prev] * dev_SupDiag[prev] + dev_x[next] * dev_SuperDiag[tid])) / dev_Diag[tid];
		dev_x[tid] = max((1-dev_om)*dev_x[tid] + dev_om * tmp, dev_payoff[tid]);
    }
	
	__syncthreads();
	
    if(tid>0 && tid < N-1 && tid%2==1){
        tmp = (dev_b[tid] - (dev_x[prev] * dev_SupDiag[prev] + dev_x[next] * dev_SuperDiag[tid])) / dev_Diag[tid];
        dev_x[tid] = max((1-dev_om)*dev_x[tid] + dev_om * tmp,dev_payoff[tid]);
    } 
	
	__syncthreads();
	
	
	tmp =0; 
	while (tid< N) {
		tmp += (dev_x[tid]-dev_y[tid])*(dev_x[tid]-dev_y[tid]);
		tid+= blockDim.x * gridDim.x;
	}
	
	cache[cacheIndex] = tmp;
	
	__syncthreads();
	
	int i=blockDim.x/2;
	while (i!=0) {
		if(cacheIndex<i)
			cache[cacheIndex] += cache[cacheIndex+i];
		
		__syncthreads();
		i/=2;
	}
	
	if(cacheIndex == 0 )
		dev_partial_sum[blockIdx.x] = cache[0];
	
}



/*************************************************************************/
/*                   FUNCTION PSOR                                       */
/*************************************************************************/
/*         Solves a linear complementary problem                         */
/*************************************************************************/

void psor(const double * Diag, const double * SuperDiag, const double * SupDiag, double * b, double * x, const double * payoff, int n, double om, int blockPerGrid){

    /* declaration */
    int  k, itmax; // iterators
	double tol;
    //double * y; 
	double * partail_sum; 
    double res;
    //double om;
	
    
    
    /* set parameters */
    itmax =1.e+6;
    //om    = 1.5;
    tol   = 1.e-11;
	//y = (double *)malloc(n*sizeof(double));	
	partail_sum = (double *)malloc(blockPerGrid*sizeof(double));
	
    double *dev_Diag;
    double *dev_SuperDiag;
    double *dev_SupDiag;
    double *dev_b;
    double *dev_payoff;
    double *dev_x;
	double *dev_y;
	double *dev_partial_sum;
    
	/* set up size for each device copy */
    int size_Diag = n*sizeof(double);
    int size_SuperDiag = n*sizeof(double);
    int size_SupDiag =	n*sizeof(double);
    int size_b = n*sizeof(double);
    int size_payoff = n*sizeof(double);
	int size_x = n*sizeof(double);
	int size_y = n*sizeof(double);
	int size_partial_sum = blockPerGrid*sizeof(double);


    /* allocate space for device copies */
    cudaMalloc((void **)&dev_Diag,size_Diag);
    cudaMalloc((void **)&dev_SuperDiag,size_SuperDiag);
    cudaMalloc((void **)&dev_SupDiag,size_SupDiag);
    cudaMalloc((void **)&dev_b,size_b);
    cudaMalloc((void **)&dev_payoff,size_payoff);
	cudaMalloc((void **)&dev_x, size_x);
	cudaMalloc((void **)&dev_y, size_y);
    cudaMalloc((void **)&dev_partial_sum, size_partial_sum);
	
	
    cudaMemcpy(dev_Diag,Diag,size_Diag,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_SuperDiag,SuperDiag,size_SuperDiag,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_SupDiag,SupDiag,size_SupDiag,cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_payoff,payoff,size_payoff,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,size_b,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_partial_sum, partail_sum, size_partial_sum,cudaMemcpyHostToDevice);
    
	//double res1;
	k = 0;
    do{
		k += 1;
		
        //memcpy(y, x, n*sizeof(double));
	
	    x[0] = max((1 - om) * x[0] + om * ( b[0] - SuperDiag[0] * x[1])/ Diag[0] , payoff[0]);
		
        
		cudaMemcpy(dev_x, x, size_x,cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y, x, size_y,cudaMemcpyHostToDevice);
		

	    psorHelper<<<blockPerGrid, threadsPerBlock>>>(dev_x,dev_y, dev_partial_sum, dev_Diag,dev_SuperDiag,dev_SupDiag,dev_b,dev_payoff,om,n);
	    
        cudaMemcpy(partail_sum, dev_partial_sum, size_partial_sum,cudaMemcpyDeviceToHost);
		cudaMemcpy(x, dev_x, size_x,cudaMemcpyDeviceToHost);
		
		//res1=0.0;
		//res1 = norm(x,y,n);
		
		res = 0;
	    for(int i=0; i< blockPerGrid; i++ ){
			res += partail_sum[i];
		}
		res = sqrt(res);
		
		//printf("res1 = %f \t res =%f \t diff = %f\n", res1, res, res1-res);
    } while (res > tol && k < itmax);
	
	
    cudaFree(dev_Diag);
    cudaFree(dev_SuperDiag);
    cudaFree(dev_SupDiag);
    cudaFree(dev_b);
    cudaFree(dev_payoff);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_partial_sum);
	
	//free(y);
	free(partail_sum);
}



