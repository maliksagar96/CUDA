#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

int main() {

  int N = 1000;
  double frequency = 1e9;
  double omega = 2 * M_PI * frequency;
  double dt = 1 / (frequency * 20);

  vector<double> f_t(N, 0);

  for (int n = 0; n < N; n++) {
    f_t[n] = sin(omega * n * dt);
  }

  vector<double> X_real(N, 0);
  vector<double> X_imag(N, 0);

  for (int k = 0; k < N; k++) {
    for (int n = 0; n < N; n++) {
      double angle = 2 * M_PI * k * n / N;
      X_real[k] += f_t[n] * cos(angle);
      X_imag[k] -= f_t[n] * sin(angle);
    }
  }

  // optional: print magnitude of non-zero bins
  for (int k = 0; k < N; k++) {
    double mag = sqrt(X_real[k]*X_real[k] + X_imag[k]*X_imag[k]);
    if (mag > 1e-6) {
      cout << "k = " << k << "  magnitude = " << mag << endl;
    }
  }

  return 0;
}
