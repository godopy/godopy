def _check_basis_data(data):
    if not issubscriptable(data):
        raise ValueError("Basis data must be subscriptable")
    if len(data) == 9:
        map(_check_numeric_scalar, data)
    elif len(data) == 3:
        map(_check_vector3_data, data)
    else:
        raise ValueError("Basis data must have 3 or 6 items")


def as_basis(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Basis(data, dtype=dtype, copy=copy)


cdef object _basis_attrs = frozenset(['rows', 'columns'])


class Basis(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        cdef numpy.ndarray base

        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if kwargs:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        if len(args) == 9:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 3:
            map(_check_vector3_data, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_basis_data(obj)
            if isinstance(args[0], numpy.ndarray) and not copy:
                if obj.dtype == dtype:
                    base = obj
                else:
                    cpp.UtilityFunctions.push_warning(
                        "Unexpected cast from %r to %r during %r initialization" % (obj.dtype, dtype, subtype)
                    )
                    base = obj.astype(dtype)
            else:
                base = np.array(obj, dtype=dtype, copy=copy)
        elif len(args) == 0:
            base = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=dtype, copy=copy)
        else:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        return PyArraySubType_NewFromBase(subtype, base)

    def __array_finalize__(self, obj):
        if isinstance(obj, Basis):
            return

        if self.shape != (3, 3):
            self.shape = (3, 3)

    def __getattr__(self, str name):
        if name == 'rows':
            return np.array(self, dtype=self.dtype, copy=False)
        elif name == 'columns':
            return np.array(self.transpose(), dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _basis_attrs:
            return object.__setattr__(self, name, value)

        if name == 'rows':
            self[:] = value
        elif name == 'columns':
            arr = value if hasattr(value, 'transpose') else np.array(value)
            self[:] = arr.transpose()
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef public object basis_to_pyobject(const cpp.Basis &b):
    cdef float [:, :] b_view = <float [:3, :3]><float *>b.rows
    cdef numpy.ndarray pyarr = Basis(b_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_basis_to_pyobject(const cpp.Variant &v):
    cdef cpp.Basis b = v.to_type[cpp.Basis]()
    cdef float [:, :] b_view = <float [:3, :3]><float *>b.rows
    cdef numpy.ndarray pyarr = Basis(b_view, dtype=np.float32, copy=True)

    return pyarr


cdef public void basis_from_pyobject(object p_obj, cpp.Basis *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (3, 3) or p_obj.dtype != np.float32:
        p_obj = as_basis(p_obj, dtype=np.float32)

    cdef cpp.Basis b
    cdef float [:, :] carr_view = <float [:3, :3]><float *>b.rows
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    carr_view[...] = pyarr_view

    r_ret[0] = b


cdef public void variant_basis_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (3, 3) or p_obj.dtype != np.float32:
        p_obj = as_basis(p_obj, dtype=np.float32)

    cdef cpp.Basis b
    cdef float [:, :] carr_view = <float [:3, :3]><float *>b.rows
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    carr_view[...] = pyarr_view

    r_ret[0] = cpp.Variant(b)
