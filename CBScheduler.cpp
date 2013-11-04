#include "CBScheduler.hpp"
#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>

using namespace std;

CBScheduler::CBScheduler(int m,int n,int numThreads){
	scheduler=new TSScheduler(m,n,numThreads);
	return;

	
	if(0)
//	if(m>=16 && n>=16 && numThreads<m && numThreads<n)
		scheduler=new InOrderScheduler(m,n,numThreads);
	else
		scheduler=new OutOfOrderScheduler(m,n,numThreads);
}

CBScheduler::~CBScheduler(){
	delete scheduler;
}

void CBScheduler::finishJob(int x,int y){
	scheduler->finishJob(x,y);
}

void CBScheduler::initScheduler(){
	scheduler->initScheduler();
}

CBScheduler::JobResult CBScheduler::getJob(int lastX,int lastY,int &newX,int &newY){
	return scheduler->getJob(lastX,lastY,newX,newY);
}


OutOfOrderScheduler::OutOfOrderScheduler(int m,int n,int numThreads):numRows(m),numColumns(n){
	numStoredRows=numColumns < numThreads+2 ? numColumns:numThreads+2;
	rowNum=new int[numStoredRows];
	isRowValid=new bool[numStoredRows];
	isRowTaken=new bool[numStoredRows];
	chessState=new BlockState *[numStoredRows];
	int i;
	for(i=0;i<numStoredRows;i++)
		chessState[i]=new BlockState[numColumns];
	isColumnTaken=new bool[numColumns];
	initScheduler();
}

OutOfOrderScheduler::~OutOfOrderScheduler(){
	delete[] rowNum;
	delete[] isRowValid;
	delete[] isRowTaken;
	delete[] isColumnTaken;
	int i;
	for(i=0;i<numStoredRows;i++)
		delete[] chessState[i];
	delete[] chessState;
}

void OutOfOrderScheduler::initScheduler(){
	int i,j;
	for(i=0;i<numStoredRows;i++){
		rowNum[i]=i;
		isRowValid[i]=true;
		isRowTaken[i]=false;
		for(j=0;j<numColumns;j++)
			chessState[i][j]=untouched;
	}
	nextRow=numStoredRows;
	for(i=0;i<numColumns;i++)
		isColumnTaken[i]=false;
}

void OutOfOrderScheduler::finishJob(int x,int y){
	int i;
	// find the row index
	for(i=0;i<numStoredRows;i++)
		if(rowNum[i]==x)
			break;
	int rowIndex=i;
	// mark the block as finished
	chessState[rowIndex][y]=finished;
	// release control of the current row and column
	isRowTaken[rowIndex]=false;
	isColumnTaken[y]=false;
	// is the row finished?
	bool rowFinished=true;
	for(i=0;i<numColumns;i++){
		if(chessState[rowIndex][i]!=finished){
			rowFinished=false;
			break;
		}
	}
	if(rowFinished){ // this row is finished, let's get a new one
		if(nextRow>=numRows){	// no new row left
			isRowValid[rowIndex]=false;
			return;
		}
		else{
			rowNum[rowIndex]=nextRow;
			nextRow++;
			for(i=0;i<numColumns;i++)
				chessState[rowIndex][i]=untouched;
		}
	}
}

CBScheduler::JobResult OutOfOrderScheduler::getJob(int lastX, int lastY, int &newX, int &newY){
	int i;
	// find the row index
	for(i=0;i<numStoredRows;i++)
		if(rowNum[i]==lastX)
			break;
	int rowIndex=i;
	bool newRow;	// is this a new row?
	if(rowIndex>=numStoredRows){
		rowIndex=0;
		newRow=true;
	}
	else
		newRow=false;
	// find a new block that shares the row with current one
	if(!newRow && isRowValid[rowIndex] && !isRowTaken[rowIndex]){
		for(i=0;i<numColumns;i++){
			if(!isColumnTaken[i] && chessState[rowIndex][i]==untouched){
				newX=lastX;
				newY=i;
				isRowTaken[rowIndex]=true;
				isColumnTaken[i]=true;
				chessState[rowIndex][i]=fetched;
				return CBScheduler::ok;
			}
		}
	}
	// find a new block that shares the column with current one
	if(!isColumnTaken[lastY]){
		for(i=0;i<numStoredRows;i++){
			if(isRowValid[i] && !isRowTaken[i] && chessState[i][lastY]==untouched){
				newX=rowNum[i];
				newY=lastY;
				isRowTaken[i]=true;
				isColumnTaken[lastY]=true;
				chessState[i][lastY]=fetched;
				return CBScheduler::ok;
			}
		}
	}
	// no block shares row or column with current one
	for(i=0;i<numStoredRows;i++){
		int j;
		if(isRowValid[i] && !isRowTaken[i]){
			for(j=0;j<numColumns;j++){
				if(!isColumnTaken[j] && chessState[i][j]==untouched){
					newX=rowNum[i];
					newY=j;
					isRowTaken[i]=true;
					isColumnTaken[j]=true;
					chessState[i][j]=fetched;
					return CBScheduler::ok;
				}
			}
		}
	}
	// no block available
	int numValidRows=0;
	int numRowsTaken=0;
	for(i=0;i<numStoredRows;i++){
		if(isRowValid[i])
			numValidRows++;
		if(isRowTaken[i])
			numRowsTaken++;
	}
	if(numRowsTaken >= numValidRows)	// all valid rows are taken
		return CBScheduler::done;
	else
		return CBScheduler::wait;
}

InOrderScheduler::InOrderScheduler(int m,int n, int _numThreads):numRows(m),numColumns(n),numBlocks(m*n),oldestBlock(0),newestBlock(-1){
	blocks=new block[numBlocks];
}

InOrderScheduler::~InOrderScheduler(){

}

void InOrderScheduler::initScheduler(){
	oldestBlock=0;
	newestBlock=-1;
	int index=0;
	for(int i=0;i<numColumns;i++){
		for(int j=0;j<numRows;j++){
			blocks[index].finished=false;
			blocks[index].x=j;
			blocks[index].y=(i+j)%numColumns;
			index++;
		}
	}
}

void InOrderScheduler::finishJob(int x,int y){
	int diff=(y-x+numColumns)%numColumns;
	int index=diff*numRows+x;
	blocks[index].finished=true;
	if(index==oldestBlock)
		while(blocks[oldestBlock].finished)
			oldestBlock++;
}

CBScheduler::JobResult InOrderScheduler::getJob(int lastX,int lastY,int &newX,int & newY){
	int windowSize=newestBlock-oldestBlock+1;
	if(newestBlock==numBlocks-1)
		return CBScheduler::done;
	if(windowSize>=numRows || windowSize>=numColumns){
		return CBScheduler::wait;
	}
	
//	int threadNum=omp_get_thread_num();
//	cout<<"threadNum: "<<threadNum<<"  "<<"newestBlock: "<<newestBlock<<endl;
//	cout<<newestBlock<<endl;
	newestBlock++;
	newX=blocks[newestBlock].x;
	newY=blocks[newestBlock].y;
	return CBScheduler::ok;
}

TwoForwardScheduler::TwoForwardScheduler(int m, int n, int _numThreads):numRows(m),numColumns(n),numThreads(_numThreads),numBlocksGet(0){
	if(m!=n){
		cerr<<"m should be equal to n in TwoForwardScheduler"<<endl;
		exit(1);
	}
	locks= new unsigned char [m];
	states=new state [m];

	for(int i=0;i<m;i++){
		locks[i]=0x00;
	}
	
	waitBit=new unsigned char[numThreads];
	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
	criticalSection=0x00;
}

TwoForwardScheduler::~TwoForwardScheduler(){
	delete[] waitBit;
	delete[] locks;
}

void TwoForwardScheduler::initScheduler(){
	numBlocksGet=0;
	blocks.clear();
	for(int i=0;i<numRows;i++)
		blocks.push_back( block(i,i) );
	numBlocksGet=numRows;

	for(int i=0;i<numRows;i++){
		states[i].x=false;
		states[i].y=0;

		locks[i]=0x00;
	}
	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
	criticalSection=0x00;
}

void TwoForwardScheduler::finishJob(int x, int y){
	int newX1,newY1;
	newX1=(x-1+numRows)%numRows;
    newY1=y;
    bool block1OK=false;
    while(testAndSet(locks[newX1]));
    if(states[newX1].x==0){
        (states[newX1].y)++;
    }
    else{
        states[newX1].x=0;
        block1OK=true;
    }
    locks[newX1]=0x00;
	
    int newX2,newY2;
    newX2=x;
    newY2=(y+1)%numRows;
    bool block2OK=false;
    while(testAndSet(locks[newX2]));
    if(states[newX2].y>0){
        states[newX2].y--;
        block2OK=true;
    }
    else
       	states[newX2].x=1;
	locks[newX2]=0x00;
	
	while(testAndSet(criticalSection));
    if(block1OK && newX1!=newY1){
        blocks.push_back(block(newX1,newY1));
		numBlocksGet++;
	}
    if(block2OK && newX2!=newY2){
        blocks.push_back(block(newX2,newY2));
		numBlocksGet++;
	}
	criticalSection=0x00;

	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
}

CBScheduler::JobResult TwoForwardScheduler::getJob(int lastX,int lastY, int & newX, int & newY){
	CBScheduler::JobResult r=CBScheduler::wait;
	int x=0;
	int y=0;
	while(r==CBScheduler::wait){
		int threadNum=omp_get_thread_num();
        while(testAndSet(waitBit[threadNum]));	
		while(testAndSet(criticalSection));
			if(!blocks.empty()){
				x=(blocks.begin())->x;
				y=(blocks.begin())->y;
				blocks.pop_front();
				r=CBScheduler::ok;
			}
			else if(numBlocksGet>=numRows*numRows){
				r=CBScheduler::done;
			}
		criticalSection=0x00;
	}
	newX=x;
	newY=y;
	return r;
}

int TwoForwardScheduler::testAndSet(unsigned char & c)
{
#ifndef WIN32
	unsigned char * addr=&c;
	unsigned char result = 1;
	asm ("xchgb %1, %0":"=m" (*addr),"=r" (result):"1" (result) :"memory");
	return result;
#else
	if(c==0x00){
		c=0x01;
		return 0;
	}
	else{
		return 1;
	}

#endif
}


TSScheduler::TSScheduler(int m,int n,int _numThreads):numRows(m),numColumns(n),numThreads(_numThreads),numBlocksGet(0){
	isRowTaken=new unsigned char[m];
	isColumnTaken=new unsigned char[n];
	blockTaken=new unsigned char *[m];
	for(int i=0;i<m;i++){
		blockTaken[i]=new unsigned char [n];
	}
	waitBit=new unsigned char [numThreads];
	for(int i=0;i<numThreads;i++){
		waitBit[i]=0x00;
	}
}

TSScheduler::~TSScheduler(){
	delete[] isRowTaken;
	delete[] isColumnTaken;
	for(int i=0;i<numRows;i++)
		delete[] blockTaken;
	delete[] blockTaken;
	delete[] waitBit;
}

void TSScheduler::initScheduler(){
	numBlocksGet=0;
	for(int i=0;i<numColumns;i++)
		isColumnTaken[i]=0x00;
	for(int i=0;i<numRows;i++){
		isRowTaken[i]=0x00;
		for(int j=0;j<numColumns;j++){
			blockTaken[i][j]=0x00;
		}
	}
	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
}

int TSScheduler::testAndSet(unsigned char & c)
{
#ifndef WIN32
	unsigned char * addr=&c;
	unsigned char result = 1;
	asm ("xchgb %1, %0":"=m" (*addr),"=r" (result):"1" (result) :"memory");
	return result;
#else
	if(c==0x00){
		c=0x01;
		return 0;
	}
	else{
		return 1;
	}

#endif
}

CBScheduler::JobResult TSScheduler::getJob(int lastX, int lastY, int &newX, int &newY){
	if(numBlocksGet>=numRows*numColumns)
		return CBScheduler::done;
	int x,y;
#ifndef WIN32
	timeval t1,t2;
	gettimeofday(&t1,NULL);
#endif	
	CBScheduler::JobResult r=CBScheduler::wait;
	if(!testAndSet(isRowTaken[lastX])){
		for(int i=0;i<numColumns;i++){	// search the blocks in the same row
			if(!testAndSet(isColumnTaken[i]))
				if(!testAndSet(blockTaken[lastX][i])){
					r=CBScheduler::ok;
					x=lastX;
					y=i;
					break;
				}
				else
					isColumnTaken[i]=0x00;
		}
		if(r==CBScheduler::ok){	// get a job
			newX=x;
			newY=y;
			#pragma omp critical(CBScheduler)
			numBlocksGet++;
			
/*			
	gettimeofday(&t2,NULL);
	cout<<"row, i="<<y<<endl;
	cout<<"time:"<<(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec<<endl;
*/			return r;
		}
		else
			isRowTaken[lastX]=0x00;
	}
	if(!testAndSet(isColumnTaken[lastY])){
		for(int i=0;i<numRows;i++){ //search the blocks in the same column
			if(!testAndSet(isRowTaken[i]))
				if(!testAndSet(blockTaken[i][lastY])){
					r=CBScheduler::ok;
					x=i;
					y=lastY;
					break;
				}
				else
					isRowTaken[i]=0x00;
		}	
		if(r==CBScheduler::ok){	// get a job
			newX=x;
			newY=y;
			#pragma omp critical(CBScheduler)
			numBlocksGet++;
/*
	cout<<"column, i="<<x<<endl;
	gettimeofday(&t2,NULL);
	cout<<"time:"<<(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec<<endl;
*/
			return r;
		}
		else
			isColumnTaken[lastY]=0x00;
	}
	for(int i=0;i<numRows;i++){
		if(!testAndSet(isRowTaken[i])){
			for(int j=0;j<numColumns;j++){	// search the blocks in the same row
				if(!testAndSet(isColumnTaken[j]))
					if(!testAndSet(blockTaken[i][j])){
						r=CBScheduler::ok;
						x=i;
						y=j;
						break;
					}
					else
						isColumnTaken[j]=0x00;
			}
			if(r==CBScheduler::ok){	// get a job
				newX=x;
				newY=y;
				#pragma omp critical(CBScheduler)
				numBlocksGet++;

/*
	gettimeofday(&t2,NULL);
	cout<<"time:"<<(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec<<endl;
	cout<<"different "<<x<<","<<y<<endl;
*/				return r;
			}
			else
				isRowTaken[i]=0x00;
		}
	}

	
/*	cout<<"no job found"<<endl;
	gettimeofday(&t2,NULL);
	cout<<"time:"<<(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec<<endl;
*/	
	//return CBScheduler::done;
	return r;
}

void TSScheduler::finishJob(int x,int y){
	isRowTaken[x]=0x00;
	isColumnTaken[y]=0x00;
	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
}

void TSScheduler::waitForChange(int threadNum){
	if(threadNum>=numThreads-1 || threadNum<0){
		return;
	}
	while( testAndSet(waitBit[threadNum]));
}



SizeScheduler::SizeScheduler(int m,int n,int _numThreads,int ** _sizeBlocks):numRows(m),numColumns(n),numThreads(_numThreads){
	isRowTaken=new bool[m];
	isColumnTaken=new bool[n];
	waitBit=new unsigned char [numThreads];

	for(int i=0;i<numRows;i++)
		for(int j=0;j<numColumns;j++){
			pair<int, block > p(_sizeBlocks[i][j], block(i,j));
			blocksInSizeOrder.push_back(p);
		}
	sort(blocksInSizeOrder.begin(),blocksInSizeOrder.end());

	initScheduler();
}

SizeScheduler::~SizeScheduler(){
	delete[] isRowTaken;
	delete[] isColumnTaken;
	delete[] waitBit;
}

void SizeScheduler::initScheduler(){
	for(int i=0;i<numColumns;i++)
		isColumnTaken[i]=false;
	for(int i=0;i<numRows;i++)
		isRowTaken[i]=false;
	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
	blocksToProcess.clear();
	for(vector< pair<int, block > >::iterator it=blocksInSizeOrder.begin();it!=blocksInSizeOrder.end();it++){
		blocksToProcess.push_back( block(it->second.x,it->second.y));
	}
}

int SizeScheduler::testAndSet(unsigned char & c)
{
#ifndef WIN32
	unsigned char * addr=&c;
	unsigned char result = 1;
	asm ("xchgb %1, %0":"=m" (*addr),"=r" (result):"1" (result) :"memory");
	return result;
#else
	if(c==0x00){
		c=0x01;
		return 0;
	}
	else{
		return 1;
	}

#endif
}

CBScheduler::JobResult SizeScheduler::getJob(int lastX, int lastY, int &newX, int &newY){
	if(blocksToProcess.empty())
		return CBScheduler::done;
	
	CBScheduler::JobResult r=CBScheduler::wait;
	#pragma omp critical(listOperation)
	{
		for(list<block>::iterator it=blocksToProcess.begin();it!=blocksToProcess.end();it++){
			int x=it->x;
			int y=it->y;
			if(!isRowTaken[x] && !isColumnTaken[y]){
				isRowTaken[x]=true;
				isColumnTaken[y]=true;
				newX=x;
				newY=y;
				blocksToProcess.erase(it);
				r=CBScheduler::ok;
				break;
			}
		}
	}
	
	return r;
}

void SizeScheduler::finishJob(int x,int y){
	isRowTaken[x]=false;
	isColumnTaken[y]=false;
	for(int i=0;i<numThreads;i++)
		waitBit[i]=0x00;
}

void SizeScheduler::waitForChange(int threadNum){
	if(threadNum>=numThreads-1 || threadNum<0){
		return;
	}
	while( testAndSet(waitBit[threadNum]));
}


bool operator < (const block & lhs,const block & rhs){
        if(lhs.x<rhs.x)
                return true;
        else if (lhs.x==rhs.x)
                return lhs.y<rhs.y;
        else
                return false;
};

