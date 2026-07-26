/*
 * pybind11 bindings for Stage 1 CUDA kernels
 *
 * This module exposes the CUDA implementations to Python.
 * Each function takes NumPy arrays as input and returns NumPy arrays.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// External declarations for CUDA functions
extern void vector_add_cuda(const float* a, const float* b, float* c, int n);
extern void matmul_cuda(const float* A, const float* B, float* C,
                        int M, int K, int N, bool use_tiled);
extern void softmax_cuda(const float* input, float* output,
                         int num_rows, int num_cols);
extern float reduce_sum_cuda(const float* input, int n);

// Vector addition wrapper
py::array_t<float> vector_add(py::array_t<float> a, py::array_t<float> b) {
    // Get buffer info
    py::buffer_info a_buf = a.request();
    py::buffer_info b_buf = b.request();

    if (a_buf.ndim != 1 || b_buf.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }

    if (a_buf.size != b_buf.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    int n = a_buf.size;

    // Create output array
    py::array_t<float> c(n);
    py::buffer_info c_buf = c.request();

    // Call CUDA function
    vector_add_cuda(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        n
    );

    return c;
}

// Matrix multiplication wrapper
py::array_t<float> matmul(py::array_t<float> A, py::array_t<float> B, bool use_tiled = true) {
    py::buffer_info A_buf = A.request();
    py::buffer_info B_buf = B.request();

    if (A_buf.ndim != 2 || B_buf.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2-dimensional");
    }

    int M = A_buf.shape[0];
    int K = A_buf.shape[1];
    int K2 = B_buf.shape[0];
    int N = B_buf.shape[1];

    if (K != K2) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }

    // Create output array
    py::array_t<float> C({M, N});
    py::buffer_info C_buf = C.request();

    // Call CUDA function
    matmul_cuda(
        static_cast<float*>(A_buf.ptr),
        static_cast<float*>(B_buf.ptr),
        static_cast<float*>(C_buf.ptr),
        M, K, N, use_tiled
    );

    return C;
}

// Softmax wrapper
py::array_t<float> softmax(py::array_t<float> input) {
    py::buffer_info in_buf = input.request();

    if (in_buf.ndim != 2) {
        throw std::runtime_error("Input must be 2-dimensional (batch x features)");
    }

    int num_rows = in_buf.shape[0];
    int num_cols = in_buf.shape[1];

    // Create output array
    py::array_t<float> output({num_rows, num_cols});
    py::buffer_info out_buf = output.request();

    // Call CUDA function
    softmax_cuda(
        static_cast<float*>(in_buf.ptr),
        static_cast<float*>(out_buf.ptr),
        num_rows, num_cols
    );

    return output;
}

// Reduce sum wrapper
float reduce_sum(py::array_t<float> input) {
    py::buffer_info in_buf = input.request();

    // Flatten array if multi-dimensional
    int n = in_buf.size;

    return reduce_sum_cuda(static_cast<float*>(in_buf.ptr), n);
}

// Module definition
PYBIND11_MODULE(cuda_ext, m) {
    m.doc() = "Stage 1 CUDA kernel implementations";

    m.def("vector_add", &vector_add,
          "Element-wise vector addition on GPU",
          py::arg("a"), py::arg("b"));

    m.def("matmul", &matmul,
          "Matrix multiplication on GPU",
          py::arg("A"), py::arg("B"), py::arg("use_tiled") = true);

    m.def("softmax", &softmax,
          "Softmax along last axis on GPU",
          py::arg("input"));

    m.def("reduce_sum", &reduce_sum,
          "Sum reduction on GPU",
          py::arg("input"));
}
