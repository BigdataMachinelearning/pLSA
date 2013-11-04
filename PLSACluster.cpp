#include <iostream>
#include <stdlib.h>
#include <string>
#include "iPLSA.hpp"
#include <fstream>
#include <omp.h>
#include <map>  
#include <vector>
#include <stdio.h>
#include <algorithm>

int cmp(const pair<int, double>& x, const pair<int, double>& y)
{
	return  x.second > y.second;	
}

void sortMapByValue(map<int, double>& tMap, vector<pair<int, double> >& tVector)
{
	for (map<int, double>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
	{ 
		tVector.push_back(make_pair(curr->first, curr->second)); 
	}	
	sort(tVector.begin(), tVector.end(), cmp);	
}

int main(int argc, char * argv[])
{
	if(argc!= 8)
	{
		cout<<"usage: PLSACluster <inputfile> <crossfolds> <numTopics> <numIters> <anneal> <numBlocks> <pos>"<<endl;
		cout<<"./PLSACluster 'data/inputtagsformat.txt' 10 200 200 100 20 0"<<endl;
		return 1;
	}

	char* inputfile=argv[1];		// input file
	int crossfold=atoi(argv[2]);	// cross validation dataset  10(1:9)
	int numLS=atoi(argv[3]);		// topic number
	int numIters=atoi(argv[4]);	// iterate number
	int anneal=atoi(argv[5]);		// simulated annealing
	int numBlocks=atoi(argv[6]);	// block number
	int pos=atoi(argv[7]);

	int cpu_core_nums = omp_get_num_procs();
	omp_set_num_threads(cpu_core_nums);
	
	iPLSA * plsa;

	plsa=new iPLSA(inputfile,crossfold, numLS, numIters, 1, 1, 0.552, anneal, 0.92, cpu_core_nums, numBlocks, pos);

	plsa->run();

	double ** p_d_z = plsa->get_p_d_z();
	double ** p_w_z = plsa->get_p_w_z();
	int document_num = plsa->numDocs();
	int topic_num = plsa->numCats();
	int word_num = plsa->numWords();

	FILE *doc2topic_fp = fopen("doc2topic_distribution.txt","w");
	if(doc2topic_fp==NULL) return -1;

	for( int i = 0; i < document_num; ++i )
	{
		for( int j = 1; j < topic_num; ++j )
		{
			fprintf(doc2topic_fp, "%f ", p_d_z[i][j]);
		}
		fprintf(doc2topic_fp, "\n");
	}

	FILE *topic2word_fp = fopen("topic2word_distribution.txt","w");
	if(doc2topic_fp==NULL) return -1;
	for( int i = 0; i < topic_num; ++i )
	{
		map<int,double> wMap;
		for( int w = 0; w<word_num; w++ )
		{
			wMap[w] = p_w_z[w][i];
		}

		vector< pair<int, double> > wVector;
		sortMapByValue(wMap,wVector);
		for( int w = 1; w<=50; w++ )
		{
			fprintf(topic2word_fp, "%d:%f ",wVector[w].first, wVector[w].second);
		}
		fprintf(topic2word_fp, "\n");
	}

	return 0;
}
