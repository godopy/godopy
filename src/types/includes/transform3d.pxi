def _check_transform3d_data(data):
    if not issubscriptable(data):
        raise ValueError("Transform3D data must be subscriptable")
    if len(data) == 12:
        map(_check_numeric_scalar, data)
    elif len(data) == 4:
        map(_check_vector3_data, data)
    elif len(data) == 2:
        _check_basis_data(data[0])
        _check_vector3_data(data[1], arg_name='origin')
    else:
        raise ValueError("Transform3D data must have 3 or 6 items")


def as_transform3d(data, dtype=None):
    if dtype is None:
        dtype = np.float32

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return Transform3D(data, dtype=dtype, copy=copy)


cdef object _transform3d_attrs = frozenset([
    'basis', 'origin'
])


class Transform3D(numpy.ndarray):
    def __new__(subtype, *args, **kwargs):
        cdef numpy.ndarray base

        dtype = kwargs.pop('dtype', np.float32)
        copy = kwargs.pop('copy', True)

        if not np.issubdtype(dtype, np.number):
            raise TypeError("%r accepts only numeric datatypes, got %r" % (subtype, dtype))

        if len(args) == 12:
            map(_check_numeric_scalar, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 4:
            map(_check_vector3_data, args)
            base = np.array(args, dtype=dtype, copy=copy)
        elif len(args) == 2:
            _check_basis_data(args[0])
            _check_vector3_data(args[1], arg_name='origin')
        elif len(args) == 1:
            obj = args[0]
            _check_transform3d_data(obj)
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
            base = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=dtype, copy=copy)
        else:
            basis = kwargs.pop('basis', None)
            origin = kwargs.pop('origin', None)

            if basis is not None and origin is None:
                raise TypeError(error_message_from_args(subtype, args, kwargs))

            _check_basis_data(basis)
            _check_vector3_data(origin, arg_name='origin')

            if len(basis) == 9:
                base = np.array([*basis, *origin], dtype=dtype, copy=copy)
            else:
                base = np.array([*basis, origin], dtype=dtype, copy=copy)

        if kwargs:
            raise TypeError(error_message_from_args(subtype, args, kwargs))

        return PyArraySubType_NewFromBase(subtype, base)

    def __array_finalize__(self, obj):
        if isinstance(obj, Transform3D):
            return

        if self.shape != (4, 3):
            self.shape = (4, 3)

    def __getattr__(self, str name):
        if name == 'basis':
            return as_basis(self[:3])
        elif name == 'origin':
            return as_vector3(self[3])

        raise AttributeError("%r has no attribute %r" % (self, name))

    def __setattr__(self, str name, object value):
        if name not in _transform3d_attrs:
            return object.__setattr__(self, name, value)

        if name == 'basis':
            self[:3] = value
        elif name == 'origin':
            self[3] = value
        else:
            raise AttributeError("%r has no attribute %r" % (self, name))


cdef public object transform3d_to_pyobject(const cpp.Transform3D &t):
    cdef float [:, :] b_view = <float [:3, :3]><float *>t.basis.rows
    cdef float [:] o_view = <float [:3]><float *>t.origin.coord
    cdef numpy.ndarray pyarr = Transform3D(dtype=np.float32)
    cdef float [:, :] pyarr_view = pyarr
    pyarr_view [:3] = b_view
    pyarr_view [3] = o_view

    return pyarr


cdef public object variant_transform3d_to_pyobject(const cpp.Variant &v):
    cdef cpp.Transform3D t = v.to_type[cpp.Transform3D]()
    cdef float [:, :] b_view = <float [:3, :3]><float *>t.basis.rows
    cdef float [:] o_view = <float [:]>t.origin.coord
    cdef numpy.ndarray pyarr = Transform3D(dtype=np.float32)
    cdef float [:, :] pyarr_view = pyarr
    pyarr_view [:3] = b_view
    pyarr_view [3] = o_view

    return pyarr


cdef public void transform3d_from_pyobject(object p_obj, cpp.Transform3D *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4, 3) or p_obj.dtype != np.float32:
        p_obj = as_transform3d(p_obj, dtype=np.float32)

    cdef cpp.Transform3D t
    cdef float [:, :] b_view = <float [:3, :3]><float *>t.basis.rows
    cdef float [:] o_view = <float [:]>t.origin.coord
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    b_view[...] = pyarr_view[:3]
    o_view[:] = pyarr_view[3]

    r_ret[0] = t


cdef public void variant_transform3d_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    if not isinstance(p_obj, numpy.ndarray) or p_obj.shape != (4, 3) or p_obj.dtype != np.float32:
        p_obj = as_transform3d(p_obj, dtype=np.float32)

    cdef cpp.Transform3D t
    cdef float [:, :] b_view = <float [:3, :3]><float *>t.basis.rows
    cdef float [:] o_view = <float [:]>t.origin.coord
    cdef float [:, :] pyarr_view = <numpy.ndarray>p_obj
    b_view[...] = pyarr_view[:3]
    o_view[:] = pyarr_view[3]

    r_ret[0] = cpp.Variant(t)
