#include <iostream>
#include <vector>
#include <cassert>
#include <ctime> 

using namespace std;

void cpuPrefixSum(vector<int>& input, vector<int>& output) {
  int currentSum = 0;
  for(int i = 0;i<input.size();i++) {
    int prefixSum = input[i] + currentSum;
    output[i] = currentSum;
    currentSum = prefixSum;
  }
}

int main() {
  int N = 10;
  vector<int> input(N), output(N);  
  srand(time(0));

  for(int i = 0;i<N;i++) {
    input[i] = rand()%10;
  }

  cpuPrefixSum(input, output);

  for(int i = 0;i<N;i++) {
    cout << "Input["<<i<<"] = "<<input[i]<<", PrefixSum["<<i<<"] = "<<output[i]<<endl;
  }

  return 0;
}