#include "darray.h"

struct darray{
	int size;
	int capacity;
	double* arr;
};

darray* darray_init(){
	darray* ret = malloc(sizeof(darray));
	if(!ret){
		return NULL;
	}
	ret->arr = malloc(sizeof(double)*7);
	if(!ret->arr){
		free(ret);
		return NULL;
	}
	ret->size=0;
	ret->capacity=7;
	return ret;
}
double* darray_get_addr(darray*m, int ind){
	return m->arr+ind;
}
double darray_get(darray* darr, int ind){
	return darr->arr[ind];
}
void darray_set(darray* darr, int ind, double val){
	darr->arr[ind]=val;
}
void darray_insert(darray* darr, double val){
	if(darr->size==darr->capacity){
		double *newarr = realloc(darr->arr,sizeof(double)*darr->capacity*2);
		if(!newarr)return;
		darr->arr=newarr;
		darr->capacity*=2;
	}
	darr->arr[darr->size]=val;
	darr->size++;
}
void darray_destroy(darray* darr){
	free(darr->arr);
	free(darr);
}

int darray_size(darray* darr){
	return darr->size;
}