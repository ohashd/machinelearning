#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

//Binary Naive Bayes
//Features are interpreted as false if 0 and true otherwise

//Uses bayes theorem:
// P(C|X) = P(x1|C)*...*P(xn|C)*P(C)/P(X)
//To calculate most likely class
//P(X) can be ignored since it is constant across all calulations.
//Also we don't really need class probaiblities, class counts will serve the same purpose
typedef struct {
	double *classCounts;
	double *featureProbabilities;
	long numOfFeatures;
	long numOfClasses;
} BNB_Model;

BNB_Model* bnb_init(BNB_Model* model,double* data, double* dependent, long rows, long cols,long numOfClasses);
void bnb_cleanup(BNB_Model* model);
long bnb_predict(BNB_Model* model,double *features);

#endif