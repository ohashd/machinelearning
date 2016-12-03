#include "gaussiannaivebayes.h"
#include <stdlib.h>
#include <math.h>

GNB_Model* gnb_init(GNB_Model* model,double* data, double* dependent, long rows, long cols,long numOfClasses){

	model->classCounts = calloc((1+2*cols)*numOfClasses,sizeof(double));

	if(!model->classCounts){
		return NULL;
	}

	model->featureMeans = model->classCounts + numOfClasses;
	model->featureVariances = model->featureMeans + numOfClasses*cols;

	model->numOfFeatures=cols;
	model->numOfClasses = numOfClasses;

	//Start Counting
	for(long r=0;r<rows;r++){
		if(dependent[r]<0 || dependent[r]>=numOfClasses) return NULL;

		long curclass = dependent[r];

		model->classCounts[curclass]++;

		for(long c=0;c<cols;c++){
				model->featureMeans[curclass*cols+c]+=data[r*cols+c];
		}

	}

	//turn counts into means
	for(long class=0;class<numOfClasses;class++){
		if(model->classCounts[class]==0)continue;
		for(long c=0;c<cols;c++){
			model->featureMeans[class*cols+c]/=model->classCounts[class];
		}
	}

	//Count deviations
	for(long r=0;r<rows;r++){
		long curclass = dependent[r];
		for(long c=0;c<cols;c++){
			double dev = model->featureMeans[curclass*cols+c]-data[r*cols+c];
			model->featureVariances[curclass*cols+c]+=dev*dev;
		}
	}

	//turn deviations into variances
	for(long class=0;class<numOfClasses;class++){
		if(model->classCounts[class]==0)continue;
		for(long c=0;c<cols;c++){
			model->featureVariances[class*cols+c]=sqrt(model->featureVariances[class*cols+c])/model->classCounts[class];
		}
	}

	return model;
}

void gnb_cleanup(GNB_Model* model){
	free(model->classCounts);
	//featureMeans and featureVariances was allocated with classCounts
}

double gnb_predict(GNB_Model* model,double *features){
	long bestClass=0;
	long cols = model->numOfFeatures;
	double bestProbability=0;
	for(long i=0; i<model->numOfClasses; i++){
		double thisprobability=model->classCounts[i];
		for(long c=0;c<cols;c++){
			if(model->featureVariances[i*cols+c]!=0){
				double featureZscore = fabs(model->featureMeans[i*cols+c]-features[c])/model->featureVariances[i*cols+c];
				thisprobability*=(1-erf(featureZscore/sqrt(2)));
			}else{
				if(model->featureMeans[i*cols+c]!=features[c]){
					thisprobability=0;break;
				}
			}
		}
		if(thisprobability>bestProbability){
			bestProbability=thisprobability;
			bestClass=i;
		}
	}
	return bestClass;
}

