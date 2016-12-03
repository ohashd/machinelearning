#include "gradientdescent.h"
#include <stdlib.h>
#include <math.h>

//Assumes data is a contiguous 2d array (1d array with size rows*cols)
//Assumes dependent is 1d array with size `rows`
GD_Model* gd_init(GD_Model* model,double *data, double* dependent, long rows,long cols, double eps){

	//+1 because we need a gradient for the constant which will be in the last slot
	long numOfWeights = cols+1;
	model->weights = calloc(numOfWeights,sizeof(double));
	if(!model->weights){
		return NULL;
	}
	//we also allocate space for errors in one go
	double*gradients = malloc((numOfWeights+rows)*sizeof(double));
	if(!gradients){
		free(model->weights);
		return NULL;
	}
	double*errors = gradients+numOfWeights;

	model->numOfFeatures = cols;

	double learningrate=0.01;
	double lastError=INFINITY;//set to infinity

	
	//cap out at 100,000 iterations
	for(long i=0;i<100000;i++){
		double totalError=0;

		//iterate through the rows and features to calculate gradients
		for(long r=0;r<rows;r++){
			//calculate what the model would predict in this row
			errors[r]=model->weights[cols];
			for(long c=0;c<cols;c++){
				errors[r]+=model->weights[c]*data[r*cols+c];
			}
			//convert to error
 			errors[r]-=dependent[r];
 			//add to totalErr
 			totalError+=errors[r]*errors[r];
 		}
		//break if error is below a threshold
		if(totalError/rows<eps)break;

 		//if we overshot, lessen the learning rate and recalulate new weights for next iteration attempt
 		if(lastError<totalError){
 			learningrate/=2;
			for(long c=0;c<numOfWeights;c++){
				model->weights[c]+=gradients[c]*2*learningrate/rows;
			}		
 			continue;
 		}
 		if(lastError>totalError)learningrate*=1.05;

 		lastError=totalError;

		//rezero gradients
		for(long c=0;c<numOfWeights;c++){
			gradients[c]=0;
		}

		for(long r=0;r<rows;r++){
			//calculate the gradient for the constant feature weight
			gradients[cols]+=errors[r];

			//calculate the gradients for the other features
			for(long c=0;c<cols;c++){
				gradients[c]+=errors[r]*data[r*cols+c];
			}
		}

		//transfer gradients into weights by a learning factor
		for(long c=0;c<numOfWeights;c++){
			model->weights[c]-=gradients[c]*2*learningrate/rows;
		}		
	}

	free(gradients);
	return model;
}

void gd_cleanup(GD_Model* model){
	free(model->weights);
}

double gd_predict(GD_Model* model, double* features){
	double ret = model->weights[model->numOfFeatures];
	for(long c=0;c<model->numOfFeatures;c++){
		ret+=model->weights[c]*features[c];
	}
	return ret;
}
