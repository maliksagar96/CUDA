#include <iostream>
#include <vector>

using namespace std;

void matrixTranspose(vector<int>& matrix, vector<int>& transpose, int nx, int ny) {
    
	for(int iy = 0; iy < ny;iy++) {
		for(int ix = 0;ix < nx;ix++) {
			transpose[ix * ny + iy] = matrix[iy * nx + ix];
		}
	}    
}

int main() {

    int nx = 4, ny = 3;

    int NM = nx*ny;

    vector<int> matrix(NM, 0);
    vector<int> transpose(NM, 0);

    for(int i = 0;i<NM;i++) {
			matrix[i] = i;
    }

    matrixTranspose(matrix, transpose, nx, ny);

    cout << "Original Matrix \n";

    for(int i = 0;i<NM;i++) {
			cout << matrix[i] << "  ";
    }

    cout << endl;

    cout << "Tranpose matrix.\n";

    for(int i = 0;i<NM;i++) {
			cout << transpose[i] << "  ";
    }

    cout << endl;


    return 0;
}