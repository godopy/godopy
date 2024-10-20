# All (4,)-shaped types are here

class Vector4(numpy.ndarray):
    pass


class Vector4i(numpy.ndarray):
    pass


class Plane(numpy.ndarray):
    pass


class Quaternion(numpy.ndarray):
    pass


class Color(numpy.ndarray):
    pass


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

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
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
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef inline numpy.ndarray array_from_rect2_args(subtype, dtype, args, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)

    if len(args) == 4:
        base = np.array(args, dtype=dtype, copy=copy)
    elif len(args) == 1:
        obj = args[0]
        if isinstance(args[0], numpy.ndarray) and not copy:
            if obj.dtype == dtype:
                base = obj
            else:
                cpp.UtilityFunctions.push_warning("Cast from %r to %r during %r initialization" % (obj.dtype, dtype, subtype))
                base = obj.astype(dtype)
        else:
            base = np.array(obj, dtype=dtype, copy=copy)
    else:
        size = args.pop() if len(args) > 1 else kwargs.pop('size', None)
        position = args.pop() if len(args) > 0 else kwargs.pop('position', None)

        if args:
            raise TypeError("Invalid positional argument %r" % args[0])
        elif kwargs:
            raise TypeError("Invalid keyword argument %r" % list(kwargs.keys).pop())

        if not issubscriptable(position) or len(position) != 2:
            raise ValueError("Invalid 'position' argument %r" % position)
        elif not issubscriptable(size) or len(size) != 2:
            raise ValueError("Invalid 'size' argument %r" % position)
        base = np.array([*position, *size], dtype=dtype, copy=copy)

    cdef numpy.ndarray ret = PyArraySubType_NewFromBase(subtype, base)

    return ret


class Rect2(_Rect2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if dtype not in (np.float32, np.float64, float):
            raise TypeError("%r accepts only 'float32' or 'float64' datatypes" % subtype)
        return array_from_rect2_args(subtype, dtype, args, kwargs)

class Rect2i(_Rect2Base):
    def __new__(subtype, *args, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if dtype not in (np.int8, np.int16, np.int32, np.int64, np.int128, int):
            raise TypeError("%r accepts only 'intX' datatypes, got %r" % (subtype, dtype))
        return array_from_rect2_args(subtype, dtype, args, kwargs)
