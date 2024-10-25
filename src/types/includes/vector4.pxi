def _check_vector4_data(data):
    if not issubscriptable(data) or (hasattr(data, 'shape') and data.shape != (4,)) or len(data) != 4:
        raise ValueError("Vector4(i) data must be a 1-dimensional container of 4 items")
    map(_check_numeric_scalar, data)


def _as_any_vector4(type_name, data, dtype, default_dtype):
    if dtype is None:
        dtype = default_dtype

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    if np.issubdtype(dtype, np.integer):
        return Vector4i(data, dtype=dtype, copy=copy, can_cast=True)
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError("%s data must be numeric" % type_name)

    return Vector4(data, dtype=dtype, copy=copy, can_cast=True)


def as_vector4(data, dtype=None):
    """
    Interpret the input as Vector4.

    If 'dtype' argument is passed interpret the result as Vector4 if dtype
    is floating or Vector4i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_vector4('Vector4', data, dtype, np.float32)


def as_vector4i(data, dtype=None):
    """
    Interpret the input as Vector4i.

    If 'dtype' argument is passed interpret the result as Vector4 if dtype
    is floating or Vector4i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_vector4('Vector4', data, dtype, np.int32)


cdef object _vector4_attrs = frozenset(['x', 'y', 'z', 'w', 'coord', 'components'])


class _Vector4Base(numpy.ndarray):
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
        if name not in _vector4_attrs:
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


cdef inline numpy.ndarray array_from_vector4_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)
    can_cast = kwargs.pop('can_cast', False)

    if kwargs:
        raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

    if args and len(args) == 4:
        map(_check_numeric_scalar, args)
        base = np.array(args, dtype=dtype)
    elif args and len(args) == 1 and issubscriptable(args[0]) and len(args[0]) == 4:
        obj = args[0]
        _check_vector4_data(obj)
        if isinstance(obj, numpy.ndarray) and not copy:
            if obj.dtype == dtype:
                base = obj
            else:
                if not can_cast:
                    cpp.UtilityFunctions.push_warning(
                        "Unexcpected cast from %r to %r during %r initialization" % (obj.dtype, dtype, subtype)
                    )
                base = obj.astype(dtype)
        else:
            base = np.array(args[0], dtype=dtype, copy=copy)
    elif len(args) == 0:
        base = np.array([0, 0, 0, 0], dtype=dtype)
    else:
        raise TypeError("%r constructor accepts only one ('coordinates'), two ('x', 'y', 'z') or no arguments" % subtype)

    return PyArraySubType_NewFromBase(subtype, base)


class Vector4(_Vector4Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)
        return array_from_vector4_args(subtype, dtype, args, kwargs)


class Vector4i(_Vector4Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes, got %r" % (subtype, dtype))
        return array_from_vector4_args(subtype, dtype, args, kwargs)


cdef public object vector4_to_pyobject(cpp.Vector4 &vec):
    cdef float [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector4(vec_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object vector4i_to_pyobject(cpp.Vector4i &vec):
    cdef int32_t [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector4i(vec_view, dtype=np.int32, copy=True)

    return pyarr


cdef public object variant_vector4_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector4 vec = v.to_type[cpp.Vector4]()
    cdef float [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector4(vec_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_vector4i_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector4i vec = v.to_type[cpp.Vector4i]()
    cdef int32_t [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector4i(vec_view, dtype=np.int32, copy=True)

    return pyarr


cdef public void vector4_from_pyobject(object p_obj, cpp.Vector4 *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4,) or p_obj.dtype != np.float32:
        p_obj = as_vector4(p_obj, dtype=np.float32)

    cdef cpp.Vector4 vec
    cdef float [:] carr_view = vec.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = vec


cdef public void vector4i_from_pyobject(object p_obj, cpp.Vector4i *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4,) or p_obj.dtype != np.int32:
        p_obj = as_vector4i(p_obj, dtype=np.int32)

    cdef cpp.Vector4i vec
    cdef int32_t [:] carr_view = vec.coord
    cdef int32_t [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = vec


cdef public void variant_vector4_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4,) or p_obj.dtype != np.float32:
        p_obj = as_vector4(p_obj, dtype=np.float32)

    cdef cpp.Vector4 vec
    cdef float [:] carr_view = vec.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(vec)


cdef public void variant_vector4i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4,) or p_obj.dtype != np.int32:
        p_obj = as_vector4i(p_obj, dtype=np.int32)

    cdef cpp.Vector4i vec
    cdef int32_t [:] carr_view = vec.coord
    cdef int32_t [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(vec)
