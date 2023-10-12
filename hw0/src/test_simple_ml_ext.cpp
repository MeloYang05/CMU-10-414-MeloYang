
#include "simple_ml_ext.hpp"

int main() {
    size_t m = 5, n = 3, k = 5;
    float* X = new float[m * n];
    float* theta = new float[n * k];
    unsigned char* y = new unsigned char[m];
    for (size_t i = 0; i < m * n; ++i) {
        X[i] = float(i + 1);
        theta[i] = float(i + 1);
    }
    for (size_t i = 0; i < m; ++i) {
        y[i] = i;
    }

    float* Z = new float[m * k];
    matrix_dot(X, theta, Z, m, n, k);
    matrix_softmax_normalize(Z, m, k);
    print_matrix(Z, m, k);
    float* Y = new float[m * k];
    vector_to_one_hot_matrix(y, Y, m, k);
    print_matrix(Y, m, k);
    float* gradients = new float(n * k);
    matrix_minus(Z, Y, m, k);
    print_matrix(Z, m, k);
    matrix_dot_trans(X, Z, gradients, n, m, k);
    print_matrix(gradients, n, k);
    matrix_div_scalar(gradients, float(m), n, k);
    matrix_mul_scalar(gradients, 0.1, n, k);
    matrix_minus(theta, gradients, n, k);
    print_matrix(gradients, n, k);
}