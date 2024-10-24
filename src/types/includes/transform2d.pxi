def _check_transform2d_data(data):
    if not issubscriptable(data):
        raise ValueError("Transform2D data must be subscriptable")
    if len(data) == 6:
        map(_check_numeric_scalar, data)
    elif len(data) == 3:
        map(_check_vector2_data, data)
    else:
        raise ValueError("Transform2D data must have 3 or 6 items")


def as_transform2d(data, dtype=0):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Transform2D(data, dtype=dtype, copy=copy, can_cast=True)


cdef object _transform2d_attrs = frozenset(['columns'])


class Transform2D(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        cdef numpy.ndarray base

        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if kwargs:
            raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

        if len(args) == 6:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 3:
            map(_check_vector2_data, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 1:
            obj = args[0]
            _check_transform2d_data(obj)
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
            base = np.array([[0, 0], [0, 0], [0, 0]], dtype=dtype, copy=copy)
        else:
            raise TypeError("Invalid positional argument %r" % args[0])

        return PyArraySubType_NewFromBase(subtype, base)

    def __array_finalize__(self, obj):
        if isinstance(obj, Transform2D):
            return

        if self.shape != (3, 2):
            self.shape = (3, 2)

    def __getattr__(self, str name):
        if name == 'columns':
            return np.array(self, dtype=self.dtype, copy=False)

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _transform2d_attrs:
            return object.__setattr__(self, name, value)

        if name == 'columns':
            self[:] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))
