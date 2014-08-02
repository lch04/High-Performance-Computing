
/*American Option Pricing using projected sor method*/

#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>



/*************************************************************************/
/*                       FUNCTION MAX                                    */
/*************************************************************************/
/*            Computes the maximal value max(a,b)                        */
/*************************************************************************/
double max(double a, double b){
  if (a >= b)
    return a;
  else
    return b;
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
/*                       FUNCTION NORM                                   */
/*************************************************************************/
/*                 Computes the tolance |a-b|/a +1                           */
/*************************************************************************/






/*************************************************************************/
/*                   FUNCTION PSOR                                       */
/*************************************************************************/
/*         Solves a linear complementary problem                         */
/*************************************************************************/
void psor(const double * Diag, const double * SuperDiag, const double * SupDiag, double * b, double * x, const double * payoff, int  n, double om){

    /* declaration */
    int i, k, itmax;
    double tol;
    double * y;
    double res;


    /* set parameters */
    itmax = 1.e+6;
    //om   = 1.2;
    tol  = 1.e-9;
    /* n = sizeof(x)/sizeof(double);*/
    /* printf("n is %d \n", n); */
    
	y = malloc(n*sizeof(double));
    
	/*double y[n];*/

    /* initialiyze algorithm, guess the x0 = MAX(A-1 * b, payoff)*/

/*
    x[0] = max(x[0] + om * b[0] / Diag[0] - SuperDiag[0] * x[1]/Diag[0], payoff[0]);
    x[n-1] = max( x[n-1] + om * b[n-1] / Diag[n-1] - SupDiag[n-2] * x[n-2]/Diag[n-1], payoff[n-1]);

    for (i = 1; i < n-1; ++i){
        x[i] = max(x[i] + om * b[i] / Diag[i] - (SupDiag[i-1] * x[i-1] - SuperDiag[i] * x[i+1])/Diag[i], payoff[i]);
    }
*/


    /* run psor algorithm */
    k = 0;
    do{double
        k = k + 1;
        memcpy(y, x, n*sizeof(double));

        x[0] = max((1 - om) * x[0] + om * ( b[0] - SuperDiag[0] * x[1])/ Diag[0] , payoff[0]);
        /* x[n-1] = max( (1 - om) * x[n-1] + om * ( b[n-1] - SupDiag[n-2] * x[n-2]) / Diag[n-1] , payoff[n-1]);*/
        for (i = 1; i < n-1; ++i){
            x[i] = max( (1 - om) * x[i] + om * ( b[i]  - SupDiag[i-1] * x[i-1] - SuperDiag[i] * x[i+1] )/Diag[i] , payoff[i]);
        }

        res = norm(x,y,n);
		//printf("res = %f \n", res);
    }while (res > tol && k < itmax);

    free(y);
}

int main(){
    double Smin =0.0;
    double Smax =100.0;
    double S =50.0;
    double rate = 0.1;
    double strike =50.0;
    double T = 5.0/12.0;
    double sigma = 0.4;
    int M =100;
    //int N =5000;

    int i=0,totalNums;
	int N[100];
	double om[100];
	//double atod ( const char * str );
	char line[100]; /* declare a char array */
	FILE *file; /* declare a FILE pointer */
	file = fopen("N.txt", "r"); /* open a text file for reading */
	while(fgets(line, sizeof line, file)!=NULL)
	{ /* keep looping until NULL pointer... */
	    N[i]=atoi(line); /* convert string to double float*/
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

    double dt = T/M;
    int j;
    for (j=15 ; j<totalNums ; j++){
    printf("N is %d, om is %0.9f\n",N[j],om[j]);
    /*printf("%f\n",dt);*/
    double ds = (Smax - Smin) / N[j];
    /*printf("%f\n",ds); */

    double M1Diag[N[j]-1];   /* diagnoal elements of A */
    double M1SuperDiag[N[j]-2]; /* superdiagonal elements of A */
    double M1SupDiag[N[j]-2]; /* supdiagonal elements of A */
    double b[N[j]-1]; /* vector in Ax=b */


    M1Diag[N[j]-2] =  1.0 + dt * (sigma * sigma * (N[j]-1 ) *(N[j]-1)+ rate );

    double sup = 0.5 * dt * (sigma * sigma * 1 - rate * 1);
    double super =  0.5 * dt * (sigma * sigma * (N[j]-1) * (N[j]-1) + rate * (N[j]-1)) ;

    for(i=1; i<N[j]-1; ++i){
        M1Diag[i-1] = 1.0 + dt * (sigma * sigma * (double) (i * i) + rate ) ;
        M1SuperDiag[i-1] = -0.5 * dt * (sigma * sigma * (double)(i * i) + rate * (double)i) ;
        M1SupDiag[i-1] = -0.5 * dt * (sigma * sigma * (i +1 )* (i+1) - rate * (i+1)) ;
    }
	M1SuperDiag[N[j]-2] = 0.0;
	M1SupDiag[N[j]-2] = 0.0;

    double payoff[N[j]-1];
    for(i=1; i< N[j]; ++i ){
        payoff[i-1] = max(strike - ds * i,0.0);
    }
    double v1 = max(strike - ds * 0.0, 0.0);
    double v2 = max(strike - ds * N[j], 0.0);

    double x[N[j]-1];
    memcpy( x, payoff, (N[j]-1)*sizeof(double));

    clock_t start = clock();
    for(i=1; i<=M; ++i){
        memcpy( b, x, (N[j]-1)*sizeof(double));
        b[0] += sup * v1;
        b[N[j]-2] += super * v2;
        /*
        for(i=0; i< N[j]-1; ++i ){
            printf("%f \n", b[i]);
        }
        */

        psor( M1Diag, M1SuperDiag, M1SupDiag, b,  x, payoff, N[j]-1, om[j]);

        /*
        for(i=0; i< N[j]-1; ++i ){
            printf("%f \n", x[i]);
        }
        */
    }
    clock_t end = clock();
    double cpu_time = (double)( end - start ) / CLOCKS_PER_SEC;

    int idx = S/ds - 1;
    printf("American Option Price is %f \n", x[idx]);
    printf("Computing time is %f \n", cpu_time);

    /*
    for(i=0; i< N[j]-1; ++i ){
        printf("%f \n", x[i]);
    }
    */
    }
    return 0;
}
