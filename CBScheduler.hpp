// chess board scheduler
#ifndef _CBScheduler_HPP
#define _CBScheduler_HPP

#include <omp.h>
#include <list>
#include <vector>
#include <map>

class SchedulerBase;

class CBScheduler{				// the interface to the outside
public:
	CBScheduler(int m,int n, int numThreads);	// create a chess board of m*n, to schedule tasks for numThreads
	~CBScheduler();

	/// initialize scheduler
	void initScheduler();
	/// finish a job
	void finishJob(int x,int y);	// finish a job at x,y
	/// JobResult
	enum JobResult{ok,wait,done};
	/// get a job
	JobResult getJob(int lastX, int lastY, int & newX, int &newY);
private:
	/// the object that does the actual job
	SchedulerBase * scheduler;	// for small chess board, use OutOfOrderScheduler, for large ones, use InOrderScheduler
};

class SchedulerBase{
public:
	/// initialize scheduler
	virtual void initScheduler()=0;
	/// finish a job
	virtual void finishJob(int x,int y)=0;	// finish a job at x,y
	/// get a job
	virtual CBScheduler::JobResult getJob(int lastX, int lastY, int & newX, int &newY)=0;
};

class InOrderScheduler:public SchedulerBase{	// in order scheduler, always schedule the blocks in the order of diagonal
public:
	InOrderScheduler(int m,int n, int numThreads);
	~InOrderScheduler();

	/// initialize scheduler
	virtual void initScheduler();
	/// finish a job
	virtual void finishJob(int x,int y);	// finish a job at x,y
	/// get a job
	virtual CBScheduler::JobResult getJob(int lastX, int lastY, int & newX, int &newY);

private:	
	int numRows;
	int numColumns;

	struct block{
		int x;
		int y;
		bool finished;
	};

	block * blocks;
	int numBlocks;
	int oldestBlock;
	int newestBlock;
};		

class OutOfOrderScheduler:public SchedulerBase{		// out of order scheduler, get any available block
public:
	OutOfOrderScheduler(int m,int n, int numThreads);
	~OutOfOrderScheduler();

	/// initialize scheduler
	virtual void initScheduler();
	/// finish a job
	virtual void finishJob(int x,int y);	// finish a job at x,y
	/// get a job
	virtual CBScheduler::JobResult getJob(int lastX, int lastY, int & newX, int &newY);

private:	
	/// size of the chess board
	int numRows;
	int numColumns;
	/// we don't have to store the whole chess board, we store numStoredRows*numClumns and drop the rows that are finished
	int numStoredRows;
	/// the stored row numbers
	int * rowNum;
	/// is a row valid?
	bool * isRowValid;
	/// the next row number
	int nextRow;
	/// is a row taken (i.e. other threads can't perform operation on that row)?
	bool * isRowTaken;
	/// is a column taken?
	bool * isColumnTaken;
	/// state of each chess block
	enum BlockState{untouched,fetched,finished};
	BlockState ** chessState;
};		


class TwoForwardScheduler:public SchedulerBase{	// in order scheduler, always schedule the blocks in the order of diagonal
public:
	TwoForwardScheduler(int m,int n, int numThreads);
	~TwoForwardScheduler();

	/// initialize scheduler
	virtual void initScheduler();
	/// finish a job
	virtual void finishJob(int x,int y);	// finish a job at x,y
	/// get a job
	virtual CBScheduler::JobResult getJob(int lastX, int lastY, int & newX, int &newY);

private:	
	int testAndSet(unsigned char & c);

	int numRows;
	int numColumns;
	int numThreads;

	struct block{
		block(int m,int n):x(m),y(n){};
		int x;
		int y;
	};

	std::list<block> blocks;
	int numBlocksGet;

	struct state{
		int x;
		int y;
	};

	state * states;
	unsigned char * locks;
	unsigned char criticalSection;
	unsigned char * waitBit;
};	


class TSScheduler:public SchedulerBase{		// out of order scheduler, get any available block
public:
	TSScheduler(int m,int n, int numThreads);
	~TSScheduler();

	/// initialize scheduler
	virtual void initScheduler();
	/// finish a job
	virtual void finishJob(int x,int y);	// finish a job at x,y
	/// get a job
	virtual CBScheduler::JobResult getJob(int lastX, int lastY, int & newX, int &newY);

	void waitForChange(int threadNum);

private:
	int testAndSet(unsigned char & c);

	/// size of the chess board
	int numRows;
	int numColumns;
	int numThreads;

	/// is a row taken
	unsigned char * isRowTaken;
	/// is a column taken?
	unsigned char * isColumnTaken;
	/// is the block taken?
	unsigned char ** blockTaken;

	int numBlocksGet;

	unsigned char * waitBit;
};		


struct block{
	block():x(0),y(0){};
	block(int m,int n):x(m),y(n){};
	int x;
	int y;
};


bool operator < (const block & lhs,const block & rhs);

class SizeScheduler:public SchedulerBase{		// out of order scheduler, get any available block
public:
	SizeScheduler(int m,int n, int numThreads,int ** sizeBlocks);
	~SizeScheduler();

	/// initialize scheduler
	virtual void initScheduler();
	/// finish a job
	virtual void finishJob(int x,int y);	// finish a job at x,y
	/// get a job
	virtual CBScheduler::JobResult getJob(int lastX, int lastY, int & newX, int &newY);

	void waitForChange(int threadNum);

private:

	int testAndSet(unsigned char & c);


	/// size of the chess board
	int numRows;
	int numColumns;
	int numThreads;

	/// is a row taken
	bool * isRowTaken;
	/// is a column taken?
	bool * isColumnTaken;

	unsigned char * waitBit;

	std::list<block> blocksToProcess;

	std::vector< std::pair<int,block> > blocksInSizeOrder;
};		
#endif
