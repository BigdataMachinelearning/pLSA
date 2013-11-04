#ifndef _IPLSA_HPP
#define _IPLSA_HPP

#include "extend_mkl.h"

#include <stdlib.h>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

class iPLSA  {
public:

	/// building without provided train/test partitions
	iPLSA(const string & matFileName, int _testPercentage, int numCats, int numIter, 
		int numRestarts, double betastart, double betastop, double anneal, double betaMod, int _numThreads, int _numBlocks, int _pos);
	virtual ~iPLSA();
    
	/// Start things going.
	void run();

	/// get the final values
	/// P(d|z) matrix
	double **get_p_d_z() const {return p_d_z_best;}
	/// P (z)
	double * get_p_z() const {return p_z_best;}
	/// P(w|z)
	double ** get_p_w_z(){return p_w_z_best;}

	// get the matrix sizes
	/// number of terms
	int numWords() const {return sizeW;}
	/// number of docs
	int numDocs() const {return sizeD;}
	/// number of categories
	int numCats() const {return sizeZ;}

private:

	// attributes
	int pos;

	/// number of threads
	int numThreads;

	/// matrix file name
	const string matFileName;

	/// number of categories
	int sizeZ;
	/// number of documents
	int sizeD;
	/// number of words
	int sizeW;

	// Row Compressed Storage of the frequency matrix
	/// two dimmensions, first dimmension is the block number
	///		freqs[1]=the freqs of the second block
	int ** freqs;
	int ** docIDs;
	int ** rowStarts;
	int ** testFreqs;
	int ** testDocIDs;
	int ** testRowStarts;

	/// start number of w, having numBlocks+1 elements
	int * wStart;
	/// start number of d
	int * dStart;

	/// Beta for TEM
	double startBeta, beta, betaMin;
	/// eta for TEM (beta = eta * beta;)
	double betaModifier;
	/// annealcue value (delta)
	double annealcue;
	/// How many iterations
	int numberOfIterations;
	/// How many restarts
	int numberOfRestarts;
	/// Best log likelihood on the test data so far.
	
	/// Percentage of data reserved for validation
	int testPercentage;

	/// P(z) vector current iteration
	double *p_z_current;
	/// P(w|z) matrix current iteration
	double **p_w_z_current;
	/// P(d|z) matrix current iteration
	double **p_d_z_current;

	/// P(z) vector previous iteration
	double *p_z_prev;
	/// P(w|z) matrix previous iteration
	double **p_w_z_prev;
	/// P(d|z) matrix previous iteration
	double **p_d_z_prev;

	/// P(z) vector best iteration
	double *p_z_best;
	/// P(w|z) matrix best iteration
	double **p_w_z_best;
	/// P(d|z) matrix best iteration
	double **p_d_z_best;

	// methods
	
	/// Load the matrix file and initialize data, dataVector, and testData
	bool loadMatrixFile();

	/// Copy current iteration data to best iteration data
	void setBestToCurrent();

	/// main routine for model training.
	void iterate();
	/// Initialize the prev probability arrays to random values.
	void initializeParameters();

	/// Initialize attributes.
	void init();

	int numBlocks;
	int numTestBlocks;

	void divideTestData(const int _numBlocks, const vector<int> * const Vectors, const vector<int> * const docIDVectors, int **& _freqs, int **& _docIDs, int **& _rowStarts);
	void divideTrainData(const int _numBlocks, const vector<int> * Vectors, const vector<int> * docIDVectors, int **& _freqs, int **& _docIDs, int **& _rowStarts, int *& _wStart, int *& _dStart);
	void testLikelihood(double & logLikelihood, double & averageLikelihood,double * p_z, double ** p_d_z, double ** p_w_z);
};
#endif /* _IPLSA_HPP */
