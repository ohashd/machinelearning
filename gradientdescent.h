#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

typedef struct {
	double *weights;
	long numOfFeatures;
} GD_Model;

GD_Model* gd_init(GD_Model* model,double* data,double* dependent, long rows, long cols, double eps);
void gd_cleanup(GD_Model* model);

double gd_predict(GD_Model* model,double *features);

#endif