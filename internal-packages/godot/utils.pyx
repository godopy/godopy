cdef extern from *:
    "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"

import numpy as np
cimport numpy as np

def initialize_numpy():
    np.import_array()
