/*
    Matrix a => dimensions = n1*n2;
    Matrib b => dimensions = n2*n3;

*/


#include <iostream>
#include <stdlib.h>

using namespace std;

void cpu_mat_mult(float *a, float *b, float *c, int n1, int n2, int n3) {

	for(int i = 0;i < n1;i++) {
		for(int j = 0;j < n2; j++) {
			float aij = a[i*n2 + j]; //a_ijth element
			for(int k = 0;k<n3;k++) {
				float bjk = b[j*n3 + k];
				//cik = summation(aij*bjk);
				c[i*n3 + k] += aij * bjk; 
			}
		}
	}

}

int main() {

	int n1 = 4, n2 = 4, n3 = 4;

	float *a = (float*)malloc(n1*n2*sizeof(float));
	float *b = (float*)malloc(n2*n3*sizeof(float));
	float *c = (float*)malloc(n1*n3*sizeof(float));

	//initialise a, b and c

	srand(1234);

	for(int i = 0;i<n1*n2;i++) {
		a[i] = (float)rand()/(float)RAND_MAX;
	}

	for(int i = 0;i<n2*n3;i++) {
		b[i] = (float)rand()/(float)RAND_MAX;
	}

	for(int i = 0;i<n1*n3;i++) {
		c[i] = 0;
	}


	cpu_mat_mult(a, b, c, n1, n2, n3);

	cout<<"**********MATRIX A**********"<<endl;
	for(int i = 0;i<n1;i++) {
		for(int j = 0;j<n2;j++) {
			cout<<a[i*n2 + j]<<"\t";
		}
		cout<<endl;
	}

	cout<<"**********MATRIX B**********"<<endl;

	for(int i = 0;i<n2;i++) {
		for(int j = 0;j<n3;j++) {
			cout<<b[i*n3 + j]<<"\t";
		}
		cout<<endl;
	}

	cout<<"**********MATRIX C**********"<<endl;
	
	for(int i = 0;i<n1;i++) {
		for(int j = 0;j<n3;j++) {
			cout<<c[i*n3 + j]<<"\t";
		}
		cout<<endl;
	}

	free(a);
	free(b);
	free(c);

	return 0;
}