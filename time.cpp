#include "time.h"
#include <sys/time.h>
#include <stddef.h>

double get_time()
{
	timeval tim;
	gettimeofday(&tim,NULL);
	return tim.tv_sec+(tim.tv_usec/1000000.0);
}
