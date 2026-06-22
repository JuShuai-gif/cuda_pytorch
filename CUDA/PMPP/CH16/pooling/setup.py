# setup.py
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "pooling_module",
        ["pooling_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # Optimization level
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="pooling_module",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
