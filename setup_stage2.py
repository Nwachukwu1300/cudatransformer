"""
Setup script for CUDA Transformer Engine - Stage 2

Builds the CUDA extension for autograd engine.
Should be run after Stage 1 setup completes successfully.
"""

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def check_cuda_available():
    """Check if CUDA toolkit is available."""
    try:
        result = subprocess.run(['nvcc', '--version'],
                                capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


CUDA_AVAILABLE = check_cuda_available()

# Extension modules (only if CUDA is available)
ext_modules = []

if CUDA_AVAILABLE:
    try:
        from pybind11.setup_helpers import Pybind11Extension

        # Find CUDA include and lib paths
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        cuda_include = os.path.join(cuda_home, 'include')
        cuda_lib = os.path.join(cuda_home, 'lib64')

        # Source files for Stage 2
        cuda_sources = [
            'stage2/cuda_ext.cpp',
            'stage2/kernels/element_wise.cu',
            'stage2/kernels/scalar_ops.cu',
            'stage2/kernels/math_ops.cu',
            'stage2/kernels/transpose.cu',
            'stage2/kernels/activations.cu',
            'stage2/kernels/reductions.cu',
            'stage2/kernels/cross_entropy.cu',
            'stage2/kernels/optimizers.cu',
        ]

        ext_modules.append(
            Pybind11Extension(
                'stage2.cuda_ext',
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
        print("pybind11 not found. Installing Python-only package.")
        CUDA_AVAILABLE = False


class CustomBuildExt(build_ext):
    """Custom build extension to handle CUDA compilation."""

    def build_extensions(self):
        if not CUDA_AVAILABLE:
            print("\n" + "=" * 60)
            print("CUDA not available. Building Python-only package.")
            print("GPU kernels will not be available.")
            print("=" * 60 + "\n")
            return

        # Check for nvcc
        try:
            self.compiler.spawn(['nvcc', '--version'])
        except Exception:
            print("nvcc not found. Skipping CUDA extension.")
            return

        super().build_extensions()


if __name__ == "__main__":
    setup(
        name='cudatransformer-stage2',
        version='0.1.0',
        description='CUDA Transformer Engine - Stage 2: Autograd Engine',
        packages=['stage2', 'stage2.kernels', 'stage2.nn', 'stage2.optim',
                  'stage2.utils', 'stage2.models', 'stage2.tests'],
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.20.0',
        ],
        extras_require={
            'dev': [
                'pytest>=7.0.0',
                'pybind11>=2.10.0',
            ],
        },
        ext_modules=ext_modules,
        cmdclass={'build_ext': CustomBuildExt},
    )
