#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

// Forward declarations of CUDA functions
extern "C" {
    // Element-wise operations
    void element_wise_multiply_cuda(const float* a, const float* b, float* c, int n);
    void element_wise_divide_cuda(const float* a, const float* b, float* c, int n, float eps);
    void element_wise_subtract_cuda(const float* a, const float* b, float* c, int n);

    // Scalar operations
    void scalar_multiply_cuda(const float* a, float scalar, float* c, int n);

    // Mathematical operations
    void exp_cuda(const float* a, float* c, int n);
    void log_cuda(const float* a, float* c, int n, float eps);
    void transpose_cuda(const float* input, float* output, int rows, int cols);

    // Activation functions
    void relu_forward_cuda(const float* input, float* output, int n);
    void relu_backward_cuda(const float* grad_output, const float* input,
                           float* grad_input, int n);
    void gelu_forward_cuda(const float* input, float* output, int n);
    void gelu_backward_cuda(const float* grad_output, const float* input,
                           float* grad_input, int n);

    // Reduction operations
    void reduce_mean_cuda(const float* input, float* output, int num_rows, int num_cols);
    void reduce_variance_cuda(const float* input, const float* mean,
                             float* output, int num_rows, int num_cols);
    void bias_add_cuda(const float* input, const float* bias,
                      float* output, int num_rows, int num_cols);

    // Cross entropy
    void cross_entropy_forward_cuda(const float* logits, const float* targets,
                                   float* loss, int num_samples, int num_classes);
    void cross_entropy_backward_cuda(const float* logits, const float* targets,
                                    float* grad_logits, int num_samples, int num_classes);

    // Optimizers
    void sgd_update_cuda(float* params, const float* grads, float learning_rate, int n);
    void adam_update_cuda(float* params, const float* grads,
                         float* m, float* v,
                         float learning_rate, float beta1, float beta2,
                         float eps, int t, int n);
}

// ================================
// Element-wise Operations
// ================================

py::array_t<float> element_wise_multiply(py::array_t<float> a, py::array_t<float> b) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != b_buf.ndim)
        throw std::runtime_error("Input arrays must have the same number of dimensions");

    int n = a_buf.size;
    if (n != b_buf.size)
        throw std::runtime_error("Input arrays must have the same total size");

    py::array_t<float> c(a_buf.size);
    auto c_buf = c.request();

    element_wise_multiply_cuda(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        n
    );

    c.resize(a_buf.shape);
    return c;
}

py::array_t<float> element_wise_divide(py::array_t<float> a, py::array_t<float> b,
                                       float eps = 1e-8f) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != b_buf.ndim)
        throw std::runtime_error("Input arrays must have the same number of dimensions");

    int n = a_buf.size;
    if (n != b_buf.size)
        throw std::runtime_error("Input arrays must have the same total size");

    py::array_t<float> c(a_buf.size);
    auto c_buf = c.request();

    element_wise_divide_cuda(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        n, eps
    );

    c.resize(a_buf.shape);
    return c;
}

py::array_t<float> element_wise_subtract(py::array_t<float> a, py::array_t<float> b) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != b_buf.ndim)
        throw std::runtime_error("Input arrays must have the same number of dimensions");

    int n = a_buf.size;
    if (n != b_buf.size)
        throw std::runtime_error("Input arrays must have the same total size");

    py::array_t<float> c(a_buf.size);
    auto c_buf = c.request();

    element_wise_subtract_cuda(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        n
    );

    c.resize(a_buf.shape);
    return c;
}

// ================================
// Scalar Operations
// ================================

py::array_t<float> scalar_multiply(py::array_t<float> a, float scalar) {
    auto a_buf = a.request();
    int n = a_buf.size;

    py::array_t<float> c(a_buf.size);
    auto c_buf = c.request();

    scalar_multiply_cuda(
        static_cast<float*>(a_buf.ptr),
        scalar,
        static_cast<float*>(c_buf.ptr),
        n
    );

    c.resize(a_buf.shape);
    return c;
}

// ================================
// Mathematical Operations
// ================================

py::array_t<float> exp(py::array_t<float> a) {
    auto a_buf = a.request();
    int n = a_buf.size;

    py::array_t<float> c(a_buf.size);
    auto c_buf = c.request();

    exp_cuda(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        n
    );

    c.resize(a_buf.shape);
    return c;
}

py::array_t<float> log(py::array_t<float> a, float eps = 1e-8f) {
    auto a_buf = a.request();
    int n = a_buf.size;

    py::array_t<float> c(a_buf.size);
    auto c_buf = c.request();

    log_cuda(
        static_cast<float*>(a_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        n, eps
    );

    c.resize(a_buf.shape);
    return c;
}

py::array_t<float> transpose(py::array_t<float> input) {
    auto input_buf = input.request();

    if (input_buf.ndim != 2)
        throw std::runtime_error("Transpose requires a 2D array");

    int rows = input_buf.shape[0];
    int cols = input_buf.shape[1];

    py::array_t<float> output({cols, rows});
    auto output_buf = output.request();

    transpose_cuda(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(output_buf.ptr),
        rows, cols
    );

    return output;
}

// ================================
// Activation Functions
// ================================

py::array_t<float> relu_forward(py::array_t<float> input) {
    auto input_buf = input.request();
    int n = input_buf.size;

    py::array_t<float> output(input_buf.size);
    auto output_buf = output.request();

    relu_forward_cuda(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(output_buf.ptr),
        n
    );

    output.resize(input_buf.shape);
    return output;
}

py::array_t<float> relu_backward(py::array_t<float> grad_output, py::array_t<float> input) {
    auto grad_output_buf = grad_output.request();
    auto input_buf = input.request();
    int n = input_buf.size;

    if (n != grad_output_buf.size)
        throw std::runtime_error("grad_output and input must have the same size");

    py::array_t<float> grad_input(input_buf.size);
    auto grad_input_buf = grad_input.request();

    relu_backward_cuda(
        static_cast<float*>(grad_output_buf.ptr),
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(grad_input_buf.ptr),
        n
    );

    grad_input.resize(input_buf.shape);
    return grad_input;
}

py::array_t<float> gelu_forward(py::array_t<float> input) {
    auto input_buf = input.request();
    int n = input_buf.size;

    py::array_t<float> output(input_buf.size);
    auto output_buf = output.request();

    gelu_forward_cuda(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(output_buf.ptr),
        n
    );

    output.resize(input_buf.shape);
    return output;
}

py::array_t<float> gelu_backward(py::array_t<float> grad_output, py::array_t<float> input) {
    auto grad_output_buf = grad_output.request();
    auto input_buf = input.request();
    int n = input_buf.size;

    if (n != grad_output_buf.size)
        throw std::runtime_error("grad_output and input must have the same size");

    py::array_t<float> grad_input(input_buf.size);
    auto grad_input_buf = grad_input.request();

    gelu_backward_cuda(
        static_cast<float*>(grad_output_buf.ptr),
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(grad_input_buf.ptr),
        n
    );

    grad_input.resize(input_buf.shape);
    return grad_input;
}

// ================================
// Reduction Operations
// ================================

py::array_t<float> reduce_mean(py::array_t<float> input) {
    auto input_buf = input.request();

    if (input_buf.ndim != 2)
        throw std::runtime_error("reduce_mean requires a 2D array");

    int num_rows = input_buf.shape[0];
    int num_cols = input_buf.shape[1];

    py::array_t<float> output({num_rows, 1});
    auto output_buf = output.request();

    reduce_mean_cuda(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(output_buf.ptr),
        num_rows, num_cols
    );

    return output;
}

py::array_t<float> reduce_variance(py::array_t<float> input, py::array_t<float> mean) {
    auto input_buf = input.request();
    auto mean_buf = mean.request();

    if (input_buf.ndim != 2)
        throw std::runtime_error("reduce_variance requires a 2D input array");

    int num_rows = input_buf.shape[0];
    int num_cols = input_buf.shape[1];

    py::array_t<float> output({num_rows, 1});
    auto output_buf = output.request();

    reduce_variance_cuda(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(mean_buf.ptr),
        static_cast<float*>(output_buf.ptr),
        num_rows, num_cols
    );

    return output;
}

py::array_t<float> bias_add(py::array_t<float> input, py::array_t<float> bias) {
    auto input_buf = input.request();
    auto bias_buf = bias.request();

    if (input_buf.ndim != 2)
        throw std::runtime_error("bias_add requires a 2D input array");
    if (bias_buf.ndim != 1)
        throw std::runtime_error("bias must be a 1D array");

    int num_rows = input_buf.shape[0];
    int num_cols = input_buf.shape[1];

    if (bias_buf.size != num_cols)
        throw std::runtime_error("bias size must match number of columns");

    py::array_t<float> output({num_rows, num_cols});
    auto output_buf = output.request();

    bias_add_cuda(
        static_cast<float*>(input_buf.ptr),
        static_cast<float*>(bias_buf.ptr),
        static_cast<float*>(output_buf.ptr),
        num_rows, num_cols
    );

    return output;
}

// ================================
// Cross Entropy Loss
// ================================

float cross_entropy_forward(py::array_t<float> logits, py::array_t<float> targets) {
    auto logits_buf = logits.request();
    auto targets_buf = targets.request();

    if (logits_buf.ndim != 2 || targets_buf.ndim != 2)
        throw std::runtime_error("logits and targets must be 2D arrays");

    int num_samples = logits_buf.shape[0];
    int num_classes = logits_buf.shape[1];

    if (targets_buf.shape[0] != num_samples || targets_buf.shape[1] != num_classes)
        throw std::runtime_error("logits and targets must have the same shape");

    float loss;
    cross_entropy_forward_cuda(
        static_cast<float*>(logits_buf.ptr),
        static_cast<float*>(targets_buf.ptr),
        &loss, num_samples, num_classes
    );

    return loss;
}

py::array_t<float> cross_entropy_backward(py::array_t<float> logits, py::array_t<float> targets) {
    auto logits_buf = logits.request();
    auto targets_buf = targets.request();

    if (logits_buf.ndim != 2 || targets_buf.ndim != 2)
        throw std::runtime_error("logits and targets must be 2D arrays");

    int num_samples = logits_buf.shape[0];
    int num_classes = logits_buf.shape[1];

    if (targets_buf.shape[0] != num_samples || targets_buf.shape[1] != num_classes)
        throw std::runtime_error("logits and targets must have the same shape");

    py::array_t<float> grad_logits({num_samples, num_classes});
    auto grad_logits_buf = grad_logits.request();

    cross_entropy_backward_cuda(
        static_cast<float*>(logits_buf.ptr),
        static_cast<float*>(targets_buf.ptr),
        static_cast<float*>(grad_logits_buf.ptr),
        num_samples, num_classes
    );

    return grad_logits;
}

// ================================
// Optimizer Updates
// ================================

void sgd_update(py::array_t<float> params, py::array_t<float> grads, float learning_rate) {
    auto params_buf = params.request();
    auto grads_buf = grads.request();

    int n = params_buf.size;
    if (n != grads_buf.size)
        throw std::runtime_error("params and grads must have the same size");

    sgd_update_cuda(
        static_cast<float*>(params_buf.ptr),
        static_cast<float*>(grads_buf.ptr),
        learning_rate, n
    );
}

void adam_update(py::array_t<float> params, py::array_t<float> grads,
                 py::array_t<float> m, py::array_t<float> v,
                 float learning_rate, float beta1, float beta2,
                 float eps, int t) {
    auto params_buf = params.request();
    auto grads_buf = grads.request();
    auto m_buf = m.request();
    auto v_buf = v.request();

    int n = params_buf.size;
    if (n != grads_buf.size || n != m_buf.size || n != v_buf.size)
        throw std::runtime_error("params, grads, m, and v must have the same size");

    adam_update_cuda(
        static_cast<float*>(params_buf.ptr),
        static_cast<float*>(grads_buf.ptr),
        static_cast<float*>(m_buf.ptr),
        static_cast<float*>(v_buf.ptr),
        learning_rate, beta1, beta2, eps, t, n
    );
}

// ================================
// Python Module Definition
// ================================

PYBIND11_MODULE(cuda_ext, m) {
    m.doc() = "Stage 2 CUDA kernel implementations for autograd engine";

    // Element-wise operations
    m.def("element_wise_multiply", &element_wise_multiply, "Element-wise multiplication");
    m.def("element_wise_divide", &element_wise_divide, "Element-wise division",
          py::arg("a"), py::arg("b"), py::arg("eps") = 1e-8f);
    m.def("element_wise_subtract", &element_wise_subtract, "Element-wise subtraction");

    // Scalar operations
    m.def("scalar_multiply", &scalar_multiply, "Scalar multiplication");

    // Mathematical operations
    m.def("exp", &exp, "Element-wise exponential");
    m.def("log", &log, "Element-wise natural logarithm",
          py::arg("a"), py::arg("eps") = 1e-8f);
    m.def("transpose", &transpose, "Matrix transpose");

    // Activation functions
    m.def("relu_forward", &relu_forward, "ReLU forward pass");
    m.def("relu_backward", &relu_backward, "ReLU backward pass");
    m.def("gelu_forward", &gelu_forward, "GELU forward pass");
    m.def("gelu_backward", &gelu_backward, "GELU backward pass");

    // Reduction operations
    m.def("reduce_mean", &reduce_mean, "Reduce mean along last axis");
    m.def("reduce_variance", &reduce_variance, "Reduce variance along last axis");
    m.def("bias_add", &bias_add, "Add bias to 2D matrix");

    // Cross entropy loss
    m.def("cross_entropy_forward", &cross_entropy_forward, "Cross entropy forward pass");
    m.def("cross_entropy_backward", &cross_entropy_backward, "Cross entropy backward pass");

    // Optimizer updates
    m.def("sgd_update", &sgd_update, "SGD parameter update (in-place)");
    m.def("adam_update", &adam_update, "Adam parameter update (in-place)",
          py::arg("params"), py::arg("grads"), py::arg("m"), py::arg("v"),
          py::arg("learning_rate"), py::arg("beta1"), py::arg("beta2"),
          py::arg("eps"), py::arg("t"));
}
