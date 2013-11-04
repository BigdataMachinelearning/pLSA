#include<iostream>
#include "omp.h"
#include <cmath>


using namespace std;


void vdPowx(int length,double *inputarray,double powx,double *outputarray)
{
     for(int i=0;i<length;i++)
    {
        outputarray[i]=pow(inputarray[i],powx);
    }
}

void vdDiv(int length,double *inputarray,double *dividend , double *outputarray)
{
    for(int i=0;i<length;i++)
        outputarray[i]=inputarray[i]/dividend[i];
}

/*
constant times a vector plus plus a vector
uses unrolled loop for increments equal to one.
*/
int daxpy(int *n , double *sa , double *sx , int *incx , double *sy ,int *incy)
{
    long int i,m,ix,iy,nn,iincx,iincy;
    register double ssa;
    nn = *n;
    ssa= *sa;
    iincx= *incx;
    iincy= *incy;

    if (nn>0 && ssa!=0.0)
    {
        if (iincx == 1 && iincy == 1)
        {
            m = nn-3;
            for (i = 0; i < m; i += 4)
            {
                sy[i] += ssa * sx[i];
                sy[i+1] += ssa * sx[i+1];
                sy[i+2] += ssa * sx[i+2];
                sy[i+3] += ssa * sx[i+3];
            }
            for ( ; i < nn; ++i)
                sy[i] += ssa * sx[i];
        }
    }
	else
    {
      ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
      iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
       for (i = 0; i < nn; i++)
      {
        sy[iy] += ssa * sx[ix];
        ix += iincx;
        iy += iincy;
      }
    }
}


double dasum(int *n, double *dx, int *incx)
{


    /* System generated locals */
    int i__1, i__2;
    double ret_val, d__1, d__2, d__3, d__4, d__5, d__6;

    /* Local variables */
    static int i, m;
    static double dtemp;
    static int nincx, mp1;


/*     takes the sum of the absolute values.   
       jack dongarra, linpack, 3/11/78.   
       modified 3/93 to return if incx .le. 0.   
       modified 12/3/93, array(1) declarations changed to array(*)   


    
   Parameter adjustments   
       Function Body */
#define DX(I) dx[(I)-1]
    ret_val = 0.;
    dtemp = 0.;
    if (*n <= 0 || *incx <= 0) {
    return ret_val;
    }
    if (*incx == 1) {
    goto L20;
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i = 1; *incx < 0 ? i >= nincx : i <= nincx; i += *incx) {
    dtemp += (d__1 = DX(i), abs(d__1));
/* L10: */
    }
    ret_val = dtemp;
    return ret_val;

/*        code for increment equal to 1   


          clean-up loop */

L20:
    m = *n % 6;
    if (m == 0) {
    goto L40;
    }
    i__2 = m;
for (i = 1; i <= m; ++i) {
    dtemp += (d__1 = DX(i), abs(d__1));
/* L30: */
    }
    if (*n < 6) {
    goto L60;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    for (i = mp1; i <= *n; i += 6) {
    dtemp = dtemp + (d__1 = DX(i), abs(d__1)) + (d__2 = DX(i + 1), abs(
        d__2)) + (d__3 = DX(i + 2), abs(d__3)) + (d__4 = DX(i + 3),
        abs(d__4)) + (d__5 = DX(i + 4), abs(d__5)) + (d__6 = DX(i + 5)
        , abs(d__6));
/* L50: */
    }
L60:
    ret_val = dtemp;
    return ret_val;
}


/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/


/* Subroutine */ 
int dscal(int *n, double *da, double *dx,int *incx)
{


    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int i, m, nincx, mp1;


/*     scales a vector by a constant.   
       uses unrolled loops for increment equal to one.   
       jack dongarra, linpack, 3/11/78.   
       modified 3/93 to return if incx .le. 0.   
       modified 12/3/93, array(1) declarations changed to array(*)   


    
   Parameter adjustments   
       Function Body */
#define DX(I) dx[(I)-1]


    if (*n <= 0 || *incx <= 0) {
	return 0;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i = 1; *incx < 0 ? i >= nincx : i <= nincx; i += *incx) {
	DX(i) = *da * DX(i);
/* L10: */
    }
    return 0;

/*        code for increment equal to 1   


          clean-up loop */

L20:
    m = *n % 5;
    if (m == 0) {
	goto L40;
    }
    i__2 = m;
    for (i = 1; i <= m; ++i) {
	DX(i) = *da * DX(i);
/* L30: */
    }
    if (*n < 5) {
	return 0;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    for (i = mp1; i <= *n; i += 5) {
	DX(i) = *da * DX(i);
	DX(i + 1) = *da * DX(i + 1);
	DX(i + 2) = *da * DX(i + 2);
	DX(i + 3) = *da * DX(i + 3);
	DX(i + 4) = *da * DX(i + 4);
/* L50: */
    }
    return 0;
} /* dscal_ */

