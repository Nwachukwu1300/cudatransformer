"""
Setup script for CUDA Transformer Engine - Stage 6 (tiled attention).

Mirrors setup.py / setup_stage2.py: builds the CUDA extension when nvcc is
available, otherwise installs nothing (the Python tiled attention in
stage6/tiled_attention.py needs no compilation and runs anywhere).

NOTE: like the Stage 1/2 setup scripts, a stock Pybind11Extension does not route
.cu files through nvcc on every platform. The most reliable way to build on
Colab is the direct nvcc command documented in stage6/COLAB_README.md (which
stage6/colab_benchmark.py runs for you). This file is provided for parity with
the rest of the repo.
"""

import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


def check_cuda_available():
    """Check if the CUDA toolkit (nvcc) is available."""
    try:
        result = subprocess.run(['nvcc', '--version'],
                                capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


CUDA_AVAILABLE = check_cuda_available()

ext_modules = []

if CUDA_AVAILABLE:
    try:
        from pybind11.setup_helpers import Pybind11Extension

        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        cuda_include = os.path.join(cuda_home, 'include')
        cuda_lib = os.path.join(cuda_home, 'lib64')

        cuda_sources = [
            'stage6/cuda_ext.cpp',
            'stage6/kernels/tiled_attention.cu',
        ]

        ext_modules.append(
            Pybind11Extension(
                'stage6.cuda_ext',
                sources=cuda_sources,
                include_dirs=[cuda_include],
                library_dirs=[cuda_lib],
                libraries=['cudart'],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': ['-O3', '--use_fast_math'],
                },
            )
        )
    except ImportError:
        print("pybind11 not found. Nothing to build.")
        CUDA_AVAILABLE = False


class CustomBuildExt(build_ext):
    """Skip the CUDA extension gracefully when nvcc is missing."""

    def build_extensions(self):
        if not CUDA_AVAILABLE:
            print("\n" + "=" * 60)
            print("CUDA not available. Skipping the Stage 6 GPU kernel.")
            print("The NumPy tiled attention still runs (no build needed).")
            print("=" * 60 + "\n")
            return
        try:
            self.compiler.spawn(['nvcc', '--version'])
        except Exception:
            print("nvcc not found. Skipping CUDA extension.")
            return
        super().build_extensions()


setup(
    name='cudatransformer-stage6',
    version='0.1.0',
    description='Stage 6: simplified tiled attention CUDA kernel',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy>=1.20.0'],
    extras_require={'dev': ['pybind11>=2.10.0']},
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExt},
)
