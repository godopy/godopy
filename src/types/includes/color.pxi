def _check_color_data(data):
    if not issubscriptable(data) \
            or (hasattr(data, 'shape') and data.shape not in ((4,), (3,))) \
            or len(data) not in (3, 4):
        raise ValueError("Color data must be a 1-dimensional container of 3 or 4 items")
    map(_check_numeric_scalar, data)


def as_color(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    # If data is an array try to reuse the data buffer
    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype or data.shape != (4,):
            copy = True
    else:
        copy = True

    return Color(data, dtype=dtype, copy=copy)


cdef object _color_attrs = frozenset(['r', 'g', 'b', 'a', 'components'])


class Color(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if kwargs:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        cdef bint data_reused = False

        if len(args) == 4:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 3:
            map(_check_numeric_scalar, args)
            base = np.array([*args, 1.], dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_color_data(obj)
            if isinstance(obj, numpy.ndarray) and not copy:
                shape = obj.shape
                if obj.dtype == dtype and shape == (4,):
                    base = obj
                    data_reused = True
                elif shape == (3,):
                    cpp.UtilityFunctions.push_warning(
                        "Unexpected reshape from (3,) to (4,) during %r initialization" % (subtype)
                    )
                    base = np.array([list(obj), 1.], dtype=dtype)
                else:
                    cpp.UtilityFunctions.push_warning(
                        "Unexpected cast from %r to %r during %r initialization" % (obj.dtype, dtype, subtype)
                    )
                    base = obj.astype(dtype)
            else:
                if len(obj) == 3:
                    obj = [*obj, 1.]
                base = np.array(obj, dtype=dtype)
        elif len(args) == 0:
            base = np.array([0., 0., 0., 0.], dtype=dtype)
        else:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        if not copy and not data_reused:
            raise ValueError("Could not reuse %r data buffer" % subtype)

        return PyArraySubType_NewFromBase(subtype, base)

    def __getattr__(self, str name):
        if name == 'r':
            return self[0]
        elif name == 'g':
            return self[1]
        elif name == 'b':
            return self[2]
        elif name == 'a':
            return self[3]
        elif name == 'components':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _color_attrs:
            return object.__setattr__(self, name, value)

        if name == 'r':
            self[0] = value
        elif name == 'g':
            self[1] = value
        elif name == 'b':
            self[2] = value
        elif name == 'a':
            self[3] = value
        elif name == 'components':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        shape = self.shape
        if shape != (4,):
            self.shape = (4,)


cdef public object color_to_pyobject(const cpp.Color &p_color):
    cdef const float [:] color_view = <float[:4]><float *>p_color.components
    cdef numpy.ndarray pyarr = as_color(color_view, dtype=np.float32)

    return pyarr


cdef public object variant_color_to_pyobject(const cpp.Variant &v):
    cdef cpp.Color color = v.to_type[cpp.Color]()
    cdef float [:] color_view = color.components
    cdef numpy.ndarray pyarr = as_color(color_view, dtype=np.float32)

    return pyarr


cdef public void color_from_pyobject(object p_obj, cpp.Color *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or not p_obj.shape == (4,) or not p_obj.dtype == np.float32:
        p_obj = as_color(p_obj, dtype=np.float32)

    cdef cpp.Color color
    cdef float [:] carr_view = color.components
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = color


cdef public void variant_color_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or not p_obj.shape == (4,) or not p_obj.dtype == np.float32:
        p_obj = as_color(p_obj, dtype=np.float32)

    cdef cpp.Color color
    cdef float [:] carr_view = color.components
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    carr_view[:] = pyarr_view

    r_ret[0] = cpp.Variant(color)
