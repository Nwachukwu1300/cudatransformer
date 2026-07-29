/*
 * pybind11 binding for the Stage 6 tiled-attention CUDA kernel.
 *
 * Exposes a single function `tiled_attention(Q, K, V, causal=True)` that takes
 * NumPy float32 arrays of shape (batch, heads, seq, head_dim) and returns the
 * attention output of the same shape, computed by the fused tiled kernel.
 *
 * Build (on a CUDA machine / Colab):
 *   python setup_stage6.py build_ext --inplace
 *   # or the direct nvcc command in stage6/COLAB_README.md
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

// Host launcher implemented in kernels/tiled_attention.cu
extern void tiled_attention_cuda(const float* hQ, const float* hK, const float* hV,
                                 float* hO, int num_bh, int seq, int head_dim, int causal);

// Force contiguous float32 so the flat (batch*heads, seq, head_dim) pointer the
// kernel indexes into matches the NumPy memory layout exactly.
using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

farray tiled_attention(farray Q, farray K, farray V, bool causal = true) {
    py::buffer_info q = Q.request();
    py::buffer_info k = K.request();
    py::buffer_info v = V.request();

    if (q.ndim != 4 || k.ndim != 4 || v.ndim != 4) {
        throw std::runtime_error("Q, K, V must be 4-D (batch, heads, seq, head_dim)");
    }
    for (int i = 0; i < 4; i++) {
        if (q.shape[i] != k.shape[i] || q.shape[i] != v.shape[i]) {
            throw std::runtime_error("Q, K, V must have identical shapes");
        }
    }

    int batch    = q.shape[0];
    int heads    = q.shape[1];
    int seq      = q.shape[2];
    int head_dim = q.shape[3];
    int num_bh   = batch * heads;   // attention is independent per (batch, head)

    // Allocate the output with the same (batch, heads, seq, head_dim) shape.
    farray O({batch, heads, seq, head_dim});
    py::buffer_info o = O.request();

    tiled_attention_cuda(
        static_cast<float*>(q.ptr),
        static_cast<float*>(k.ptr),
        static_cast<float*>(v.ptr),
        static_cast<float*>(o.ptr),
        num_bh, seq, head_dim, causal ? 1 : 0);

    return O;
}

PYBIND11_MODULE(cuda_ext, m) {
    m.doc() = "Stage 6 tiled (FlashAttention-style) attention CUDA kernel";
    m.def("tiled_attention", &tiled_attention,
          "Fused tiled scaled-dot-product attention on GPU (online softmax)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal") = true);
}
