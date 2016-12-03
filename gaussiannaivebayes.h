#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

//Gaussian Naive Bayes

typedef struct{
	double *classCounts;
	double *featureMeans;
	double *featureVariances;
	long numOfFeatures;
	long numOfClasses;
} GNB_Model;

GNB_Model* gnb_init(GNB_Model* model,double* data, double* dependent, long rows, long cols,long numOfClasses);
void gnb_cleanup(GNB_Model* model);
double gnb_predict(GNB_Model* model,double *features);

#endif