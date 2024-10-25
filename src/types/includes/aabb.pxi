def _check_aabb_data(data):
    if not issubscriptable(data):
        raise ValueError("AABB data must be subscriptable")
    if len(data) == 6:
        map(_check_numeric_scalar, data)
    elif len(data) == 2:
        _check_vector3_data(data[0], arg_name='position')
        _check_vector3_data(data[1], arg_name='size')
    else:
        raise ValueError("AABB data must have 2 or 6 items")


def as_aabb(data, dtype=0):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return AABB(data, dtype=dtype, copy=copy, can_cast=True)


cdef frozenset _aabb_attrs = frozenset([
    'x', 'y', 'z', 'sx', 'sy', 'sz', 'position', 'size_'
])


class AABB(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        cdef numpy.ndarray base

        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)
        dtype = kwargs.pop('dtype', np.float32)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes" % subtype)

        if len(args) == 6:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 2:
            _check_vector3_data(args[0], arg_name='position')
            _check_vector3_data(args[1], arg_name='size')
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_aabb_data(obj)
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
            base = np.array([[0, 0, 0], [0, 0, 0]], dtype=dtype, copy=copy)
        else:
            position = kwargs.pop('position', None)
            size = kwargs.pop('size', None) or kwargs.pop('size_', None)

            if position is not None and size is None:
                # No valid keyword arguments, therefore something wrong with positional args
                raise TypeError("Invalid positional argument %r" % args[0])

            _check_vector3_data(position, arg_name='position')
            _check_vector3_data(size, arg_name='size')

            base = np.array([position, size], dtype=dtype, copy=copy)

        if kwargs:
            raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

        return PyArraySubType_NewFromBase(subtype, base)

    def __getattr__(self, str name):
        if name == 'x':
            return self[0][0]
        elif name == 'y':
            return self[0][1]
        elif name == 'z':
            return self[0][2]
        elif name == 'sx':
            return self[1][0]
        elif name == 'sy':
            return self[1][1]
        elif name == 'sz':
            return self[1][2]
        elif name == 'position':
            if np.issubdtype(self.dtype, np.integer):
                return Vector3i(self[0], dtype=self.dtype, copy=False)
            else:
                return Vector3(self[0], dtype=self.dtype, copy=False)
        elif name == 'size_':
            if np.issubdtype(self.dtype, np.integer):
                return Vector3i(self[1], dtype=self.dtype, copy=False)
            else:
                return Vector3(self[1], dtype=self.dtype, copy=False)
        else:
            raise AttributeError("%s has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _rect2_attrs:
            return object.__setattr__(self, name, value)

        if name == 'x':
            self[0][0] = value
        elif name == 'y':
            self[0][1] = value
        elif name == 'z':
            self[0][2] = value
        elif name == 'sx':
            self[1][0] = value
        elif name == 'sy':
            self[1][1] = value
        elif name == 'sz':
            self[1][2] = value
        elif name == 'position':
            self[0] = value
        elif name == 'size_':
            self[1] = value
        else:
            raise AttributeError("%s has no attribute %r" % (self, name))

    def __array_finalize__(self, obj):
        if isinstance(obj, AABB):
            return

        if self.shape != (2, 3):
            self.shape = (2, 3)


cdef public object aabb_to_pyobject(cpp._AABB &p_aabb):
    cdef float [:] position_view = p_aabb.position.coord
    cdef float [:] size_view = p_aabb.size.coord

    cdef numpy.ndarray pyarr = AABB([position_view, size_view], dtype=np.float32, copy=True)

    return pyarr


cdef public object variant_aabb_to_pyobject(const cpp.Variant &v):
    cdef cpp._AABB aabb = v.to_type[cpp._AABB]()

    cdef float [:] position_view = aabb.position.coord
    cdef float [:] size_view = aabb.size.coord

    cdef numpy.ndarray pyarr = AABB([position_view, size_view], dtype=np.float32, copy=True)

    return pyarr


cdef public void aabb_from_pyobject(object p_obj, cpp._AABB *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (2, 3) or p_obj.dtype != np.float32:
        p_obj = as_aabb(p_obj, dtype=np.float32)

    cdef cpp._AABB aabb
    cdef float [:] position_view = aabb.position.coord
    cdef float [:] size_view = aabb.size.coord

    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    position_view[:] = pyarr_view[0]
    size_view[:] = pyarr_view[1]

    r_ret[0] = aabb


cdef public void variant_aabb_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (2, 3) or p_obj.dtype != np.float32:
        p_obj = as_aabb(p_obj, dtype=np.float32)

    cdef cpp._AABB aabb
    cdef float [:] position_view = aabb.position.coord
    cdef float [:] size_view = aabb.size.coord
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    position_view[:] = pyarr_view[0]
    size_view[:] = pyarr_view[1]

    r_ret[0] = cpp.Variant(aabb)
