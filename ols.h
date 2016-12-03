#ifndef OLS_H
#define OLS_H

typedef struct {
	double *weights;
	long numOfFeatures;
} OLS_Model;

OLS_Model* ols_init(OLS_Model* model,double* data, double* dependent, long rows, long cols);
void ols_cleanup(OLS_Model* model);

double ols_predict(OLS_Model* model,double *features);

#endif