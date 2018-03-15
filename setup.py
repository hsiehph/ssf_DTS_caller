import os

from distutils.core import setup, Extension 
from Cython.Distutils import build_ext
from Cython.Build import cythonize 

import numpy as np

numpy_path = os.path.dirname(np.__file__)
python_path = numpy_path.replace("site-packages/numpy", "")

ext_modules = [
               Extension("traverse_contours", 
                         ["traverse_contours.pyx"],
                         language="c",
                         include_dirs = ["%s/core/include/" % numpy_path]
                         ),
               Extension("get_windowed_variance", 
                         ["get_windowed_variance.pyx"],
                         language="c",
                         include_dirs = ["%s/core/include/" % numpy_path]
                         ),
               Extension("c_hierarchical_edge_merge",
                         ['c_hierarchical_edge_merge.pyx',
                         './heap/heap.cc',
                         './edge.cc',
                         './float_heap.cc'],
                        language = "c++",
                        include_dirs = [python_path, "heap/"]
                        )
              ]

setup( 
    cmdclass = {'build_ext':build_ext},
    name = 'c_SSF',
    ext_modules = ext_modules 
    )
