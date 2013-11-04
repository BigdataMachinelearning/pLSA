#include "Schedule.hpp"
 
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>

void Schedule::setAffinity(int cpu_id)
{
	const pid_t pid = getpid();
 
#ifdef LIB_01
	cpu_set_t cpu_mask;
	CPU_ZERO(&cpu_mask);
	CPU_SET(cpu_id, &cpu_mask);
 
	sched_setaffinity(0, &cpu_mask);
#endif
 
#ifdef LIB_02
	unsigned long cpu_mask = (unsigned long) 1;
	cpu_mask=cpu_mask<<cpu_id; 
	sched_setaffinity(0, sizeof(unsigned long), &cpu_mask);
#endif
 
#ifdef LIB_03
	cpu_set_t cpu_mask;
	CPU_ZERO(&cpu_mask);
	CPU_SET(cpu_id, &cpu_mask);
 
	sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask);
#endif
}

