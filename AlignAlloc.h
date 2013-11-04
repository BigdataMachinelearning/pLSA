#ifndef _ALIGN_ALLOC_H
#define _ALIGN_ALLOC_H

using namespace std;

class AlignAlloc
{
	public:
		static char *al_alloc(unsigned long size);
		static void al_free(void *p);
};

#endif
