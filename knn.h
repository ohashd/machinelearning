#ifndef KNN_H
#define KNN_H

typedef struct {
	double *data;
	double *dependent;
	long rows;
	long cols;
	long k;
} KNN_Model;

KNN_Model* knn_init(KNN_Model* model,double* data,double * dependent, long rows, long cols, long k);
void knn_cleanup(KNN_Model* model);

double knn_predict(KNN_Model* model,double *features);

#endif