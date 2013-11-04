#ifndef EXTEND_MKL_H
#define EXTEND_MKL_H

#include<iostream>
#include "omp.h"
#include <cmath>
using namespace std;

void vdPowx(int length,double *inputarray,double powx,double *outputarray);
int daxpy(int *n , double *sa , double *sx , int *incx , double *sy ,int *incy);
double dasum(int *n, double *dx, int *incx);
void vdDiv(int length,double *inputarray,double *dividend , double *outputarray);
int dscal(int *n, double *da, double *dx,int *incx);
#endif /*EXTEND_MKL_H*/
