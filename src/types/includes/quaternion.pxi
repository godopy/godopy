def _check_quaternion_data(data):
    if not issubscriptable(data) or (hasattr(data, 'shape') and data.shape != (4,)) or len(data) != 4:
        raise ValueError("Quaternion data must be a 1-dimensional container of 4 items")
    map(_check_numeric_scalar, data)


def as_quaternion(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Quaternion(data, dtype=dtype, copy=copy)


cdef object _quaternion_attrs = frozenset(['x', 'y', 'z', 'w', 'components', 'coord'])


class Quaternion(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if kwargs:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        if len(args) == 4:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_quaternion_data(obj)
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
            base = np.array([0, 0, 0, 0], dtype=dtype, copy=copy)
        else:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        return PyArraySubType_NewFromBase(subtype, base)

    def __getattr__(self, str name):
        if name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'z':
            return self[2]
        elif name == 'w':
            return self[3]
        elif name == 'coord' or name == 'components':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _quaternion_attrs:
            return object.__setattr__(self, name, value)

        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'z':
            self[2] = value
        elif name == 'w':
            self[3] = value
        elif name == 'coord' or name == 'components':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        shape = self.shape
        if shape != (4,):
            self.shape = (4,)


cdef public object quaternion_to_pyobject(const cpp.Quaternion &q):
    cdef const float [:] q_view = <float[:4]><float *>q.components
    cdef numpy.ndarray pyarr = Quaternion(q_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_quaternion_to_pyobject(const cpp.Variant &v):
    cdef cpp.Quaternion q = <cpp.Quaternion>v
    cdef float [:] q_view = q.components
    cdef numpy.ndarray pyarr = Quaternion(q_view, dtype=np.float32, copy=True)

    return pyarr


cdef public void quaternion_from_pyobject(object p_obj, cpp.Quaternion *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or not p_obj.shape == (4,) or not p_obj.dtype == np.float32:
        p_obj = as_quaternion(p_obj, dtype=np.float32)

    cdef cpp.Quaternion q
    cdef float [:] carr_view = q.components
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = q


cdef public void variant_quaternion_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or not p_obj.shape == (4,) or not p_obj.dtype == np.float32:
        p_obj = as_quaternion(p_obj, dtype=np.float32)

    cdef cpp.Quaternion q
    cdef float [:] carr_view = q.components
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(q)
