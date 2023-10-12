#ifndef CMU_10_414_HW0_HPP
#define CMU_10_414_HW0_HPP

#include <memory.h>

#include <cmath>
#include <iostream>

void print_matrix(float* A, size_t m, size_t n);

void matrix_dot(const float* A, const float* B, float* C, size_t m, size_t n, size_t k);

void matrix_dot_trans(const float* A, const float* B, float* C, size_t m, size_t n, size_t k);

void matrix_minus(float* A, const float* B, size_t m, size_t n);

void matrix_mul_scalar(float* C, float scalar, size_t m, size_t n);

void matrix_div_scalar(float* C, float scalar, size_t m, size_t n);

void matrix_softmax_normalize(float* C, size_t m, size_t n);

void vector_to_one_hot_matrix(const unsigned char* y, float* Y, size_t m, size_t k);

#endif