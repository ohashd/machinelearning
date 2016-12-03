#include "knn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

KNN_Model* knn_init(KNN_Model* model,double* data, double *dependent,long rows, long cols, long k){
	model->data = malloc(sizeof(double)*rows*(cols+1));
	if(!model->data){
		return NULL;
	}
	memcpy(model->data,data,sizeof(double)*cols*rows);

	model->dependent = model->data+cols*rows;
	memcpy(model->dependent,dependent,sizeof(double)*rows);

	model->rows = rows;
	model->cols = cols;
	model->k = k;
	return model;
}
void knn_cleanup(KNN_Model* model){
	free(model->data);
}

double knn_predict(KNN_Model* model,double *features){
	struct{
		double prediction;
		double distance;
	}* closestRows = malloc(sizeof(double)*2*model->k);
	
	if(!closestRows){
		return NAN;
	}

	for(long k=0;k<model->k;k++){
		closestRows[k].distance=INFINITY;
	}

	long numOfFeatures = model->cols;

	for(long r = 0;r<model->rows;r++){
		double dist = 0;
		for(long c =0;c<numOfFeatures;c++){
			double ldist = (features[c]-model->data[r*numOfFeatures+c]);
			dist+= ldist*ldist;
		}

		//If we have a perfect match, return. (Also prevents dividing by 0 later)
		if(dist==0) return model->dependent[r];

		//check if this row is closer to features (first item in closestRows will always be the most distant closest point)
		if(dist>=closestRows[0].distance)continue;

		long k=1;
		while(k<model->k && dist<closestRows[k].distance) k++;
		k--;
		//move all elements back
		for(long i=0;i<k;i++){
			closestRows[i]=closestRows[i+1];
		}
		
		closestRows[k].prediction = model->dependent[r];
		closestRows[k].distance = dist;

	}


	double prediction=0;
	double totalWeights=0;

	//make all distances inversely relative to the largest distance
	//(ie a row half as far as the largest distance will have twice the weight)
	closestRows[0].distance=sqrt(closestRows[0].distance);
	for(long k=1;k<model->k;k++){
		closestRows[k].distance=closestRows[0].distance/sqrt(closestRows[k].distance);
		totalWeights+=closestRows[k].distance;
	}
	closestRows[0].distance=1;
	totalWeights+=1;

	for(long k=0;k<model->k;k++){
		prediction+=closestRows[k].prediction*closestRows[k].distance;
	}
	prediction/=totalWeights;

	return prediction;
}