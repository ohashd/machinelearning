#include <stdio.h>
#include <errno.h>
#include "gradientdescent.h"
#include "knn.h"
#include "darray.h"
#include "ols.h"
#include "gaussiannaivebayes.h"

int main(){
	int numOfCols=1;
	int numOfRows=0;
	GD_Model gd_model;
	KNN_Model knn_model;
	OLS_Model ols_model;
	GNB_Model gnb_model;
	double *features=0;
	darray* a = darray_init();
	darray* b = darray_init();
	char buff[1024];

	//open file
	FILE *fp = fopen("testdata.csv", "r");
	if(!fp){
		printf("Error opening file.");
		goto exit;
	};

	//Get number of columns
	if(!fgets(buff,1024,fp))goto exit;
	for(char*c=buff; *c!=0;c++){
		if(*c==',')numOfCols++;
	}
	numOfCols--;
	do{/*read a line*/
		char *p=buff;
		for(;;){/* Process Numbers */
			double temp = strtod(p,&p);

			if(*p==','){ /*Check if read succeeded*/
				darray_insert(a,temp);
				p++;//skip the comma
			}else if(*p=='\n'||*p==0){
				darray_insert(b,temp);
				break;//read next line
			}else{
				printf("Error converting to double\n");
				goto exit;			
			}
		}
		numOfRows++;
	}while(fgets(buff,1024,fp));

	if(ferror(fp)){
		printf("Error reading\n");
		goto exit;
	}
	//check number of entries is correct
	if(numOfRows*numOfCols!=darray_size(a)){
		printf("Error: number of data entries is not number of rows x number of columns.");
		goto exit;
	}

	//initiate gradient descent model
	if(!gd_init(&gd_model,
		darray_get_addr(a,0),
		darray_get_addr(b,0),
		numOfRows,numOfCols,0.001)){
		printf("Error allocating memory for gradient descent model");
	}
	//initiate knn model
	if(!knn_init(&knn_model,
		darray_get_addr(a,0),
		darray_get_addr(b,0),
		numOfRows,numOfCols,2)){
		printf("Error allocating memory for knn model");
	}

	//initiate ols model
	if(!ols_init(&ols_model,
		darray_get_addr(a,0),
		darray_get_addr(b,0),
		numOfRows,numOfCols)){
		printf("Error allocating memory for ols model");
	}
	//initiate gnb model
	if(!gnb_init(&gnb_model,
		darray_get_addr(a,0),
		darray_get_addr(b,0),
		numOfRows,numOfCols,3)){
		printf("Error allocating memory for gnb model");
	}

	features = malloc(sizeof(double)*(numOfCols));


	//use the model to predict from stdin
	while(fgets(buff,1024,stdin)){
		char *p=buff;
		for(int i=0;i<numOfCols;i++){/* Process Numbers */
			double temp = strtod(p,&p);
			if(*p==','){ /*Check if read succeeded*/
				features[i]=temp;
				p++;//skip the comma
			}else if(*p=='\n'){
				features[i]=temp;
				break;//read next line
			}else{
				printf("Error converting to double\n");
				goto exit;			
			}
		}
		printf("%f\n",gd_predict(&gd_model,features));
		printf("%f\n",knn_predict(&knn_model,features));
		printf("%f\n",ols_predict(&ols_model,features));
		printf("%f\n",gnb_predict(&gnb_model,features));
	}

	if(ferror(stdin)){
		printf("Error reading from stdin.");
		goto exit;
	}

exit:
	free(features);
	gd_cleanup(&gd_model);
	knn_cleanup(&knn_model);
	ols_cleanup(&ols_model);
	gnb_cleanup(&gnb_model);
	darray_destroy(a);
	if(fp)fclose(fp);	
	return 0;
}