def _check_projection_data(data):
    if not issubscriptable(data):
        raise ValueError("Projection data must be subscriptable")
    if len(data) == 16:
        map(_check_numeric_scalar, data)
    elif len(data) == 4:
        map(_check_vector4_data, data)
    else:
        raise ValueError("Projection data must have 3 or 6 items")


def as_projection(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Projection(data, dtype=dtype, copy=copy)


cdef object _projection_attrs = frozenset(['columns', 'rows'])


class Projection(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        cdef numpy.ndarray base

        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if kwargs:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        if len(args) == 16:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 4:
            map(_check_vector4_data, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_projection_data(obj)
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
            base = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=dtype, copy=copy)
        else:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        return PyArraySubType_NewFromBase(subtype, base)

    def __array_finalize__(self, obj):
        if isinstance(obj, Projection):
            return

        if self.shape != (4, 4):
            self.shape = (4, 4)

    def __getattr__(self, str name):
        if name == 'columns':
            return np.array(self, dtype=self.dtype, copy=False)
        elif name == 'rows':
            return np.array(self.transpose(), dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _projection_attrs:
            return object.__setattr__(self, name, value)

        if name == 'columns':
            self[:] = value
        elif name == 'rows':
            arr = value if hasattr(value, 'transpose') else np.array(value)
            self[:] = arr.transpose()
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef public object projection_to_pyobject(const cpp.Projection &p):
    cdef float [:, :] p_view = <float [:4, :4]><float *>p.columns
    cdef numpy.ndarray pyarr = as_projection(p_view, dtype=np.float32)

    return pyarr


cdef public object variant_projection_to_pyobject(const cpp.Variant &v):
    cdef cpp.Projection p = <cpp.Projection>v
    cdef float [:, :] p_view = <float [:4, :4]><float *>p.columns
    cdef numpy.ndarray pyarr = as_projection(p_view, dtype=np.float32)

    return pyarr


cdef public void projection_from_pyobject(object p_obj, cpp.Projection *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4, 4) or p_obj.dtype != np.float32:
        p_obj = as_projection(p_obj, dtype=np.float32)

    cdef cpp.Projection p
    cdef float [:, :] carr_view = <float [:4, :4]><float *>p.columns
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    carr_view[...] = pyarr_view

    r_ret[0] = p


cdef public void variant_projection_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4, 4) or p_obj.dtype != np.float32:
        p_obj = as_projection(p_obj, dtype=np.float32)

    cdef cpp.Projection p
    cdef float [:, :] carr_view = <float [:4, :4]><float *>p.columns
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    carr_view[...] = pyarr_view

    r_ret[0] = cpp.Variant(p)
