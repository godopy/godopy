def _check_plane_data(data):
    if not issubscriptable(data):
        raise ValueError("Plane data must be subscriptable")
    elif hasattr(data, 'shape') and data.shape != (4,):
        raise ValueError("Plane data must be a 1-dimensional container of 4 items"
                         "or a 1-dimensional container of 3 items and a scalar")
    elif len(data) == 4:
        map(_check_numeric_scalar, data)
    elif len(data) == 2:
        _check_vector3_data(data[0], arg_name='normal')
        _check_numeric_scalar(data[1], arg_name='d')
    else:
        raise ValueError("Plane data must have 4 or 2 items")


def as_plane(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Plane(data, dtype=dtype, copy=copy, can_cast=True)


cdef object _plane_attrs = frozenset(['normal', 'd', 'x', 'y', 'z'])


class Plane(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if len(args) == 4:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 2:
            _check_vector3_data(args[0], arg_name='normal')
            _check_numeric_scalar(args[1], arg_name='d')
            base = np.array([*args[0], args[1]], dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_plane_data(obj)
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
            normal = kwargs.pop('normal', None)
            d = kwargs.pop('d', None)

            if normal is not None and d is None:
                # No valid keyword arguments, therefore something wrong with positional args
                raise TypeError("Invalid positional argument %r" % args[0])

            _check_vector3_data(normal, arg_name='normal')
            _check_numeric_scalar(d, arg_name='d')
            base = np.array([*normal, d], dtype=dtype, copy=copy)

        if kwargs:
            raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

        return PyArraySubType_NewFromBase(subtype, base)

    def __getattr__(self, str name):
        if name == 'normal':
            return self[:3]
        elif name == 'd':
            return self[3]
        elif name == 'x':
            return self[0]
        elif name == 'y':
            return self[1]
        elif name == 'z':
            return self[2]

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _vector4_attrs:
            return object.__setattr__(self, name, value)

        if name == 'normal':
            self[:3] = value
        elif name == 'd':
            self[3] = value
        elif name == 'x':
            self[0] = value
        elif name == 'y':
            self[1] = value
        elif name == 'z':
            self[2] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        if isinstance(obj, Plane):
            return

        if self.shape != (4,):
            self.shape = (4,)


cdef public object plane_to_pyobject(cpp.Plane &plane):
    cdef numpy.ndarray pyarr = Plane([*list(plane.normal.coord), plane.d], dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_plane_to_pyobject(const cpp.Variant &v):
    cdef cpp.Plane plane = v.to_type[cpp.Plane]()
    cdef numpy.ndarray pyarr = Plane([*list(plane.normal.coord), plane.d], dtype=np.float32, copy=True)

    return pyarr


cdef public void plane_from_pyobject(object p_obj, cpp.Plane *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or not p_obj.shape == (4,) or not p_obj.dtype == np.float32:
        p_obj = as_plane(p_obj, dtype=np.float32)

    cdef cpp.Plane plane
    cdef float [:] normal_view = plane.normal.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    normal_view[:] = pyarr_view[:3]
    plane.d = pyarr_view[4]

    r_ret[0] = plane


cdef public void variant_plane_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or not p_obj.shape == (4,) or not p_obj.dtype == np.float32:
        p_obj = as_plane(p_obj, dtype=np.float32)

    cdef cpp.Plane plane
    cdef float [:] normal_view = plane.normal.coord
    cdef float [:] pyarr_view = <numpy.ndarray>p_obj
    normal_view[:] = pyarr_view[:3]
    plane.d = pyarr_view[4]

    r_ret[0] = cpp.Variant(plane)
