#ifndef __DARRAY__
#define __DARRAY__

#include <stdlib.h>

typedef struct darray darray;

darray* darray_init();
double* darray_get_addr(darray*, int ind);
double darray_get(darray*, int ind);
void darray_set(darray*, int ind, double val);
void darray_insert(darray*, double val);
void darray_destroy(darray*);
int darray_size(darray*);


#endif