def _check_vector3_data(data, arg_name=None):
    if not issubscriptable(data) or (hasattr(data, 'shape') and data.shape != (3,)) or len(data) != 3:
        argument = 'argument %r' % arg_name if arg_name else 'data'
        raise ValueError("Vector3(i) %s must be a 1-dimensional container of 3 items" % argument)
    map(_check_numeric_scalar, data)


def _as_any_vector3(type_name, data, dtype, default_dtype):
    if dtype is None:
        dtype = default_dtype

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    if np.issubdtype(dtype, np.integer):
        return Vector3i(data, dtype=dtype, copy=copy, can_cast=True)
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError("%s data must be numeric" % type_name)

    return Vector3(data, dtype=dtype, copy=copy, can_cast=True)


def as_vector3(data, dtype=None):
    """
    Interpret the input as Vector3.

    If 'dtype' argument is passed interpret the result as Vector3 if dtype
    is floating or Vector3i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_vector3('Vector3', data, dtype, np.float32)


def as_vector3i(data, dtype=None):
    """
    Interpret the input as Vector3i.

    If 'dtype' argument is passed interpret the result as Vector3 if dtype
    is floating or Vector3i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_vector3('Vector3', data, dtype, np.int32)


cdef object _vector3_attrs = frozenset(['x', 'y', 'z', 'coord', 'components'])


class _Vector3Base(numpy.ndarray):
    def __getattr__(self, str name):
        if name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'z':
            return self[2]
        elif name == 'coord' or name == 'components':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _vector3_attrs:
            return object.__setattr__(self, name, value)

        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'z':
            self[2] = value
        elif name == 'coord' or name == 'components':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        shape = self.shape
        if shape != (3,):
            self.shape = (3,)


cdef inline numpy.ndarray array_from_vector3_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)
    can_cast = kwargs.pop('can_cast', False)

    if kwargs:
        raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

    if args and len(args) == 3:
        map(_check_numeric_scalar, args)
        base = np.array(args, dtype=dtype)
    elif args and len(args) == 1 and issubscriptable(args[0]) and len(args[0]) == 4:
        obj = args[0]
        _check_vector3_data(obj)
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
        base = np.array([0, 0, 0], dtype=dtype)
    else:
        raise TypeError("%r constructor accepts only one ('coordinates'), two ('x', 'y', 'z') or no arguments" % subtype)

    return PyArraySubType_NewFromBase(subtype, base)


class Vector3(_Vector3Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)
        return array_from_vector3_args(subtype, dtype, args, kwargs)


class Vector3i(_Vector3Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes, got %r" % (subtype, dtype))
        return array_from_vector3_args(subtype, dtype, args, kwargs)


cdef public object vector3_to_pyobject(cpp.Vector3 &vec):
    cdef float [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector3(vec_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object vector3i_to_pyobject(cpp.Vector3i &vec):
    cdef int32_t [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector3i(vec_view, dtype=np.int32, copy=True)

    return pyarr


cdef public object variant_vector3_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector3 vec = v.to_type[cpp.Vector3]()
    cdef float [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector3(vec_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_vector3i_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector3i vec = v.to_type[cpp.Vector3i]()
    cdef int32_t [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = Vector3i(vec_view, dtype=np.int32, copy=True)

    return pyarr


cdef public void vector3_from_pyobject(object p_obj, cpp.Vector3 *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (3,) or p_obj.dtype != np.float32:
        p_obj = as_vector3(p_obj, dtype=np.float32)

    cdef cpp.Vector3 vec
    cdef float [:] carr_view = vec.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = vec


cdef public void vector3i_from_pyobject(object p_obj, cpp.Vector3i *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (3,) or p_obj.dtype != np.int32:
        p_obj = as_vector3i(p_obj, dtype=np.int32)

    cdef cpp.Vector3i vec
    cdef int32_t [:] carr_view = vec.coord
    cdef int32_t [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = vec


cdef public void variant_vector3_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (3,) or p_obj.dtype != np.float32:
        p_obj = as_vector3(p_obj, dtype=np.float32)

    cdef cpp.Vector3 vec
    cdef float [:] carr_view = vec.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(vec)


cdef public void variant_vector3i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (3,) or p_obj.dtype != np.int32:
        p_obj = as_vector3i(p_obj, dtype=np.int32)

    cdef cpp.Vector3i vec
    cdef int32_t [:] carr_view = vec.coord
    cdef int32_t [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(vec)
