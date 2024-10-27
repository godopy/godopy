def _check_vector2_data(data, arg_name=None):
    if not issubscriptable(data) or (hasattr(data, 'shape') and data.shape != (2,)) or len(data) != 2:
        argument = 'argument %r' % arg_name if arg_name else 'data'
        raise ValueError("Vector2(i) %s must be a 1-dimensional container of 2 items" % argument)
    map(_check_numeric_scalar, data)


def _as_any_vector2(type_name, data, dtype, default_dtype):
    if dtype is None:
        dtype = default_dtype

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    if np.issubdtype(dtype, np.integer):
        return Vector2i(data, dtype=dtype, copy=copy)
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError("%s data must be numeric" % type_name)

    return Vector2(data, dtype=dtype, copy=copy)


def as_vector2(data, dtype=None):
    """
    Interpret the input as Vector2.

    If 'dtype' argument is passed interpret the result as Vector2 if dtype
    is floating or Vector2i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_vector2('Vector2', data, dtype, np.float32)


def as_vector2i(data, dtype=None):
    """
    Interpret the input as Vector2i.

    If 'dtype' argument is passed interpret the result as Vector2 if dtype
    is floating or Vector2i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_vector2('Vector2', data, dtype, np.int32)


cdef object _vector2_attrs = frozenset(['x', 'y', 'coord', 'components'])
cdef object _size2_attrs = frozenset(['width', 'height', 'x', 'y', 'coord', 'components'])


class _Vector2Base(numpy.ndarray):
    def __getattr__(self, str name):
        if name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'coord' or name == 'components':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _vector2_attrs:
            return object.__setattr__(self, name, value)

        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'coord':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        shape = self.shape
        if shape != (2,):
            self.shape = (2,)


class _Size2Base(_Vector2Base):
    def __getattr__(self, str name):
        if name == 'width' or name == 'x':
            return self[0]
        elif name == 'height' or name == 'y':
            return self[1]
        elif name == 'coord' or name == 'components':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _size2_attrs:
            return object.__setattr__(self, name, value)

        if name == 'width' or name == 'x':
            self[0] = value
        elif name == 'height' or name == 'y':
            self[1] = value
        elif name == 'coord' or name == 'components':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef inline numpy.ndarray array_from_vector2_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)

    if kwargs:
        raise TypeError(error_message_from_args(subtype, args, kwargs))

    if args and len(args) == 2:
        map(_check_numeric_scalar, args)
        base = np.array(args, dtype=dtype)
    elif args and len(args) == 1 and issubscriptable(args[0]):
        obj = args[0] 
        _check_vector2_data(obj)
        if isinstance(obj, numpy.ndarray) and not copy:
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
        base = np.array([0, 0], dtype=dtype)
    else:
        raise TypeError(error_message_from_args(subtype, args, kwargs))

    return PyArraySubType_NewFromBase(subtype, base)


class Vector2(_Vector2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)
        return array_from_vector2_args(subtype, dtype, args, kwargs)


class Size2(_Size2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)
        return array_from_vector2_args(subtype, dtype, args, kwargs)


class Vector2i(_Vector2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes, got %r" % (subtype, dtype))
        return array_from_vector2_args(subtype, dtype, args, kwargs)


class Size2i(_Size2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes, got %r" % (subtype, dtype))
        return array_from_vector2_args(subtype, dtype, args, kwargs)



cdef public object vector2_to_pyobject(const cpp.Vector2 &vec):
    cdef const float [:] vec_view = <float[:2]><float *>vec.coord
    cdef numpy.ndarray pyarr = as_vector2(vec_view, dtype=np.float32, copy=True)

    return pyarr


cdef public object vector2i_to_pyobject(const cpp.Vector2i &vec):
    cdef const int32_t [:] vec_view = <int32_t[:2]><int32_t *>vec.coord
    cdef numpy.ndarray pyarr = as_vector2i(vec_view, dtype=np.int32)

    return pyarr


cdef public object variant_vector2_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector2 vec = v.to_type[cpp.Vector2]()
    cdef const float [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = as_vector2(vec_view, dtype=np.float32)

    return pyarr


cdef public object variant_vector2i_to_pyobject(const cpp.Variant &v):
    cdef cpp.Vector2i vec = v.to_type[cpp.Vector2i]()
    cdef const int32_t [:] vec_view = vec.coord
    cdef numpy.ndarray pyarr = as_vector2i(vec_view, dtype=np.int32)

    return pyarr


cdef public void vector2_from_pyobject(object p_obj, cpp.Vector2 *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (2,) or p_obj.dtype != np.float32:
        p_obj = as_vector2(p_obj, dtype=np.float32)

    cdef cpp.Vector2 vec
    cdef float [:] carr_view = vec.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = vec


cdef public void vector2i_from_pyobject(object p_obj, cpp.Vector2i *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (2,) or p_obj.dtype != np.int32:
        p_obj = as_vector2i(p_obj, dtype=np.int32)

    cdef cpp.Vector2i vec
    cdef int32_t [:] carr_view = vec.coord
    cdef int32_t [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = vec


cdef public void variant_vector2_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (2,) or p_obj.dtype != np.float32:
        p_obj = as_vector2(p_obj, dtype=np.float32)

    cdef cpp.Vector2 vec
    cdef float [:] carr_view = vec.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(vec)


cdef public void variant_vector2i_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (2,) or p_obj.dtype != np.int32:
        p_obj = as_vector2i(p_obj, dtype=np.int32)

    cdef cpp.Vector2i vec
    cdef int32_t [:] carr_view = vec.coord
    cdef int32_t [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(vec)
