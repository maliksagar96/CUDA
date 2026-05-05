/*
    In parallel sum reduction of lets say an array size of 16. 
    1) We add adjacent elements (0,1),(2,3),(4,5),(6,7),(7,8)... . We save them in elements 0,2,4,6,8
    2) Then we add (0,2), (4,6) and so on. 

    Following program is a CPU implementation of the code. To better understand the redcution approach.
*/

#include <iostream>
#include <vector>

using namespace std;

void cpu_implementation(vector<int>& h_v) {

    int sz = h_v.size();

    //Strided for loop. Doubling s every iteration. stride will go like 1,2,4,8,16,32...
    for(int stride = 1;stride < sz; stride *= 2) {
        //i will go like 0,2,4,6,8 ... in the first iteration. 
        //i will go like 0, 4, 8, 12, 16 ... in the second iteration.
        for(int i = 0;i<sz-stride;i+=2*stride) {
            h_v[i] = h_v[i] + h_v[i+stride];
        }
    }

    cout << h_v[0] << endl;
}

int main() {

	int N = 1 << 16;    //Vector size is 2^16.
	vector<int> h_v(N,1);    
    cpu_implementation(h_v);

    return 0;
}