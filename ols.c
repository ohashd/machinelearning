#include "ols.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//OLS Estimation with QR Decomposition
OLS_Model* ols_init(OLS_Model* model,double* data,double* dependent, long rows, long cols){

	long numOfWeights = cols+1;

	double *R = calloc(numOfWeights*numOfWeights,sizeof(double));
	if(!R){
		return NULL;
	}
	//working in Q transpose improves cache performance
	double *Qt = malloc(sizeof(double)*numOfWeights*rows);
	if(!Qt){
		free(R);
		return NULL;
	}

	model->weights = calloc(numOfWeights,sizeof(double));
	if(!model->weights){
		free(R);free(Qt);
		return NULL;
	}

	model->numOfFeatures = cols;

	//setup first row as the constant row
	for(long r=0;r<rows;r++){
		Qt[r]=1;
	}
	//Transpose data into Qt offset by first row
	for(long c=1;c<numOfWeights;c++){
		for(long r=0;r<rows;r++){
				Qt[c*rows+r]=data[r*cols+c-1];
		}
	}
	//Perform QR Decom
	for(long c=0;c<numOfWeights;c++){//loop through each column vector
		//Calculate vector length
		double val=0;
		double vectorlength = 0;
		for(long r=0;r<rows;r++){
			val = Qt[c*rows+r];
			vectorlength += val*val; 
		}
		vectorlength=sqrt(vectorlength);
		//if column vector has no significance we can skip it
		if(vectorlength<0.0000001) continue;

		//set diagonal value in R
		R[c*numOfWeights+c]=vectorlength;

		//Normalize vector
		for(long r=0;r<rows;r++){
			Qt[c*rows+r] /=vectorlength;
		}

		//Update other vectors in R and Qt
		for(long k=c+1;k<numOfWeights;k++){
			//Calculate k column (dot) c column
			double dotproduct=0;
			for(long r=0;r<rows;r++){
				dotproduct+= Qt[k*rows+r]*Qt[c*rows+r];
			}
			//Set R appropriately for Q vector c and Original vector k
			R[c*numOfWeights+k]=dotproduct;
			//Make kth vector orthogonal to vector c
			for(long r=0;r<rows;r++){
				Qt[k*rows+r]-=Qt[c*rows+r]*R[c*numOfWeights+k];
			}
		}
	}


	//Solve System of Linear Equations. [Qt][dependent] = [R][model->weights]
	for(long c=cols;c>=0;c--){

		//Skip insignificant vectors
		//We can compare with zero here because values close to zero get filtered above
		//when checking for vector length <0.000001
		if(R[c*numOfWeights+c]==0)continue;

		double qtdependent=0;
		for(long r=0;r<rows;r++){
			qtdependent+=Qt[c*rows+r]*dependent[r];
		}


		//Remove all known variables
		for(long k=cols;k>c;k--){
			qtdependent-=model->weights[k]*R[c*numOfWeights+k];
		}


		model->weights[c]=qtdependent/R[c*numOfWeights+c];
	}

	free(R);
	free(Qt);
	return model;
}

void ols_cleanup(OLS_Model* model){
	free(model->weights);
}

double ols_predict(OLS_Model* model,double *features){
	double ret = model->weights[0];
	for(long c=1;c<=model->numOfFeatures;c++){
		ret+=model->weights[c]*(*features++);
	}
	return ret;
}