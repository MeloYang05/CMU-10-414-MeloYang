#include "simple_ml_ext.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void print_matrix(float* A, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float* A, const float* B, float* C, size_t m, size_t n, size_t k) {
    memset(C, 0, m * k * sizeof(float));
    size_t A_row = 0, B_row = 0, C_row = 0;
    float a = 0;
    for (size_t i = 0; i < m; ++i) {
        B_row = 0;
        for (size_t h = 0; h < n; ++h) {
            a = A[A_row + h];
            for (size_t j = 0; j < k; ++j) {
                C[C_row + j] += a * B[B_row + j];
            }
            B_row += k;
        }
        A_row += n;
        C_row += k;
    }
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float* A, const float* B, float* C, size_t m, size_t n, size_t k) {
    memset(C, 0, m * k * sizeof(float));
    size_t A_row = 0, B_row = 0, C_row = 0;
    float a = 0;
    for (size_t i = 0; i < n; ++i) {
        C_row = 0;
        for (size_t j = 0; j < m; ++j) {
            a = A[A_row + j];
            for (size_t h = 0; h < k; ++h) {
                C[C_row + h] += a * B[B_row + h];
            }
            C_row += k;
        }
        A_row += m;
        B_row += k;
    }
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float* A, const float* B, size_t m, size_t n) {
    size_t A_size = m * n;
    for (size_t i = 0; i < A_size; ++i) {
        A[i] -= B[i];
    }
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float* C, float scalar, size_t m, size_t n) {
    size_t C_size = m * n;
    for (size_t i = 0; i < C_size; ++i) {
        C[i] *= scalar;
    }
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float* C, float scalar, size_t m, size_t n) {
    size_t C_size = m * n;
    for (size_t i = 0; i < C_size; ++i) {
        C[i] /= scalar;
    }
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float* C, size_t m, size_t n) {
    size_t C_size = m * n;
    // Double to store very large floats
    double sum = 0;
    double* exp_values = new double[n];
    for (size_t i = 0; i < C_size; i += n) {
        sum = 0;
        for (size_t j = 0; j < n; ++j) {
            exp_values[j] = exp(C[i + j]);
            sum += exp_values[j];
        }
        for (size_t j = 0; j < n; ++j) {
            C[i + j] = exp_values[j] / sum;
        }
    }
    delete[] exp_values;
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char* y, float* Y, size_t m, size_t k) {
    size_t Y_size = m * k, j = 0;
    memset(Y, 0, Y_size * sizeof(float));
    for (size_t i = 0; i < m; ++i) {
        Y[j + y[i]] = 1;
        j += k;
    }
}

void softmax_regression_epoch_cpp(const float* X, const unsigned char* y,
                                  float* theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */
    float* Z = new float[batch * k];
    float* Y = new float[batch * k];
    float* gradients = new float[n * k];
    size_t m_b = batch;
    for (size_t offset = 0; offset < m; offset += batch) {
        const float* X_b = X + offset * n;
        const unsigned char* y_b = y + offset;
        m_b = m - offset > batch ? batch : m - offset;
        matrix_dot(X_b, theta, Z, m_b, n, k);
        matrix_softmax_normalize(Z, m_b, k);
        vector_to_one_hot_matrix(y_b, Y, m_b, k);
        matrix_minus(Z, Y, m_b, k);
        matrix_dot_trans(X_b, Z, gradients, n, m_b, k);
        matrix_div_scalar(gradients, float(m_b), n, k);
        matrix_mul_scalar(gradients, lr, n, k);
        matrix_minus(theta, gradients, n, k);
    }
    delete[] Z;
    delete[] Y;
    delete[] gradients;
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
            softmax_regression_epoch_cpp(
                static_cast<const float*>(X.request().ptr),
                static_cast<const unsigned char*>(y.request().ptr),
                static_cast<float*>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
