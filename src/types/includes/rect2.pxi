def _check_rect2_data(data):
    if not issubscriptable(data):
        raise ValueError("Rect2(i) data must be subscriptable")
    if len(data) == 4:
        map(_check_numeric_scalar, data)
    elif len(data) == 2:
        _check_vector2_data(data[0], arg_name='position')
        _check_vector2_data(data[1], arg_name='size')
    else:
        raise ValueError("Rect2(i) data must have 2 or 4 items")


def _as_any_rect2(type_name, data, dtype, default_dtype):
    if dtype is None:
        dtype = default_dtype

    copy = False

    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    if np.issubdtype(dtype, np.integer):
        return Rect2i(data, dtype=dtype, copy=copy, can_cast=True)
    elif not np.issubdtype(dtype, np.floating):
        raise ValueError("%s data must be numeric" % type_name)

    return Rect2(data, dtype=dtype, copy=copy, can_cast=True)


def as_rect2(data, dtype=None):
    """
    Interpret the input as Rect2.

    If 'dtype' argument is passed interpret the result as Rect2 if dtype
    is floating or Rect2i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_rect2('Rect2', data, dtype, np.float32)


def as_rect2i(data, dtype=None):
    """
    Interpret the input as Rect2i.

    If 'dtype' argument is passed interpret the result as Rect2 if dtype
    is floating or Rect2i if dtype is integer. Non-numeric dtypes will
    raise a ValueError.
    """
    return _as_any_rect2('Rect2i', data, dtype, np.int32)


cdef frozenset _rect2_attrs = frozenset([
    'x', 'y', 'width', 'height', 'position', 'size_', 'coord'
])

class _Rect2Base(numpy.ndarray):
    def __getattr__(self, str name):
        if name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'width':
            return self[2]
        elif name == 'height':
            return self[3]
        elif name == 'position':
            if np.issubdtype(self.dtype, np.integer):
                return Vector2i(self[:2], dtype=self.dtype, copy=False)
            else:
                return Vector2(self[:2], dtype=self.dtype, copy=False)
        elif name == 'size_':
            if np.issubdtype(self.dtype, np.integer):
                return Size2i(self[2:], dtype=self.dtype, copy=False)
            else:
                return Size2(self[2:], dtype=self.dtype, copy=False)
        elif name == 'coord':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%s has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _rect2_attrs:
            return object.__setattr__(self, name, value)

        if name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'width':
            self[2] = value
        elif name == 'height':
            self[3] = value
        elif name == 'position':
            self[:2] = value
        elif name == 'size_':
            self[2:] = value
        elif name == 'coord':
            self[:] = value
        else:
            raise AttributeError("%s has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        if isinstance(obj, _Rect2Base):
            return

        if self.shape != (4,):
            self.shape = (4,)


cdef inline numpy.ndarray array_from_rect2_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)
    can_cast = kwargs.pop('can_cast', False)

    if len(args) == 4:
        map(_check_numeric_scalar, args)
        base = np.array(args, dtype=dtype, copy=copy)
    elif len(args) == 2:
        _check_vector2_data(args[0], arg_name='position')
        _check_vector2_data(args[1], arg_name='size')
        base = np.array(args, dtype=dtype, copy=copy)
    elif len(args) == 1:
        obj = args[0]
        _check_rect2_data(obj)
        if isinstance(args[0], numpy.ndarray) and not copy:
            if obj.dtype == dtype:
                base = obj
            else:
                if not can_cast:
                    cpp.UtilityFunctions.push_warning(
                        "Unexpected cast from %r to %r during %r initialization" % (obj.dtype, dtype, subtype)
                    )
                base = obj.astype(dtype)
        else:
            base = np.array(obj, dtype=dtype, copy=copy)
    elif len(args) == 0:
        base = np.array([0, 0, 0, 0], dtype=dtype, copy=copy)
    else:
        position = kwargs.pop('position', None)
        size = kwargs.pop('size', None) or kwargs.pop('size_', None)

        if position is not None and size is None:
            # No valid keyword arguments, therefore something wrong with positional args
            raise TypeError("Invalid positional argument %r" % args[0])

        _check_vector2_data(position, arg_name='position')
        _check_vector2_data(size, arg_name='size')

        base = np.array([position, size], dtype=dtype, copy=copy)

    if kwargs:
        raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

    return PyArraySubType_NewFromBase(subtype, base)


class Rect2(_Rect2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)

        return array_from_rect2_args(subtype, dtype, args, kwargs)

class Rect2i(_Rect2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes, got %r" % (subtype, dtype))

        return array_from_rect2_args(subtype, dtype, args, kwargs)


cdef public object rect2_to_pyobject(cpp.Rect2 &rect):
    cdef float [:] position_view = rect.position.coord
    cdef float [:] size_view = rect.size.coord

    cdef numpy.ndarray pyarr = Rect2([position_view, size_view], dtype=np.float32, copy=True)

    return pyarr


cdef public object rect2i_to_pyobject(cpp.Rect2i &rect):
    cdef int32_t [:] position_view = rect.position.coord
    cdef int32_t [:] size_view = rect.size.coord

    cdef numpy.ndarray pyarr = Rect2i([position_view, size_view], dtype=np.int32, copy=True)

    return pyarr


cdef public object variant_rect2_to_pyobject(const cpp.Variant &v):
    cdef cpp.Rect2 rect = v.to_type[cpp.Rect2]()
    cdef float [:] position_view = rect.position.coord
    cdef float [:] size_view = rect.size.coord

    cdef numpy.ndarray pyarr = Rect2([position_view, size_view], dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_rect2i_to_pyobject(const cpp.Variant &v):
    cdef cpp.Rect2i rect = v.to_type[cpp.Rect2i]()
    cdef int32_t [:] position_view = rect.position.coord
    cdef int32_t [:] size_view = rect.size.coord

    cdef numpy.ndarray pyarr = Rect2i([position_view, size_view], dtype=np.int32, copy=True)

    return pyarr


cdef public void rect2_from_pyobject(object obj, cpp.Rect2 *r_ret) noexcept:
    cdef cpp.Rect2 rect
    cdef float [:] position_view = rect.position.coord
    cdef float [:] size_view = rect.size.coord

    carr_view_from_pyobject[float [:]](obj, position_view, np.float32, 4, 0, 2)
    carr_view_from_pyobject[float [:]](obj, size_view, np.float32, 4, 2)

    r_ret[0] = rect


cdef public void rect2i_from_pyobject(object obj, cpp.Rect2i *r_ret) noexcept:
    cdef cpp.Rect2i rect
    cdef int32_t [:] position_view = rect.position.coord
    cdef int32_t [:] size_view = rect.size.coord

    carr_view_from_pyobject[int32_t [:]](obj, position_view, np.float32, 4, 0, 3)
    carr_view_from_pyobject[int32_t [:]](obj, size_view, np.float32, 4, 2)

    r_ret[0] = rect


cdef public void variant_rect2_from_pyobject(object obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.Rect2 rect
    cdef float [:] position_view = rect.position.coord
    cdef float [:] size_view = rect.size.coord

    carr_view_from_pyobject[float [:]](obj, position_view, np.float32, 4, 0, 2)
    carr_view_from_pyobject[float [:]](obj, size_view, np.float32, 4, 2)

    r_ret[0] = cpp.Variant(rect)


cdef public void variant_rect2i_from_pyobject(object obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.Rect2i rect
    cdef int32_t [:] position_view = rect.position.coord
    cdef int32_t [:] size_view = rect.size.coord

    cdef int _
    carr_view_from_pyobject[int32_t [:]](obj, position_view, np.float32, 4, 0, 3)
    carr_view_from_pyobject[int32_t [:]](obj, size_view, np.float32, 4, 2)

    r_ret[0] = cpp.Variant(rect)
