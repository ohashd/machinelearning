#include "naivebayes.h"
#include <stdlib.h>

BNB_Model* bnb_init(BNB_Model* model,double* data, double* dependent, long rows, long cols,long numOfClasses){

	model->classCounts = calloc((1+cols)*numOfClasses,sizeof(double));

	if(!model->classCounts){
		return NULL;
	}

	model->featureProbabilities = model->classCounts + numOfClasses;
	model->numOfFeatures=cols;
	model->numOfClasses = numOfClasses;

	//Start Counting
	for(long r=0;r<rows;r++){
		if(dependent[r]<0 || dependent[r]>=numOfClasses) return NULL;

		long curclass = dependent[r];

		model->classCounts[curclass]++;

		for(long c=0;c<cols;c++){
			if(data[r*cols+c]!=0){
				model->featureProbabilities[curclass*cols+c]++;
			}
		}

	}

	//turn counts into probabilities
	for(long class=0;class<numOfClasses;class++){
		if(model->classCounts[class]==0)continue;
		for(long c=0;c<cols;c++){
			model->featureProbabilities[class*cols+c]/=model->classCounts[class];
		}
	}

	return model;
}

void bnb_cleanup(BNB_Model* model){
	free(model->classCounts);
	//featureProbabilities was allocated with classCounts
}

long bnb_predict(BNB_Model* model,double *features){
	long bestClass=0;
	double bestProbability=0;
	for(long i=0; i<model->numOfClasses; i++){
		double thisprobability=model->classCounts[i];
		for(long c=0;c<model->numOfFeatures;c++){
			if(features[c]!=0){
				thisprobability*=model->featureProbabilities[i*model->numOfFeatures+c];
			}else{
				thisprobability*=(1-model->featureProbabilities[i*model->numOfFeatures+c]);
			}
		}
		if(thisprobability>bestProbability){
			bestProbability=thisprobability;
			bestClass=i;
		}
	}
	return bestClass;
}