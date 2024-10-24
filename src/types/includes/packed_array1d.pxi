def _as_any_1d_array(type_name, data, dtype, default_dtype):
    if dtype is None:
        dtype = default_dtype
    if not issubscriptable(data):
        raise ValueError("%s data must be subscriptable" % type_name)

    cdef int itemsize = -1

    # Try not to copy here
    copy = False

    if isinstance(data, numpy.ndarray):
        itemsize = data.itemsize
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    if np.issubdtype(dtype, np.floating):
        if itemsize > 2:
            return PackedFloat64Array(data, dtype=dtype, copy=copy, can_cast=True)
        else:
            return PackedFloat32Array(data, dtype=dtype, copy=copy, can_cast=True)
    elif isstring_dtype(dtype):
        return PackedStringArray(data, dtype=dtype, copy=copy, can_cast=True)
    elif not np.issubdtype(dtype, np.integer):
        raise ValueError("Packed<Type>Array data must be numeric or string-like, "
                         "requested dtype (%s) is not." % dtype)

    if itemsize > 1:
        if itemsize > 2:
            return PackedInt64Array(data, dtype=dtype, copy=copy, can_cast=True)
        else:
            return PackedInt32Array(data, dtype=dtype, copy=copy, can_cast=True)

    return PackedByteArray(data, dtype=dtype, copy=copy, can_cast=True)


def as_packed_byte_array(data, dtype=None):
    """
    Interpret the input as PackedByteArray.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_1d_array('PackedByteArray', data, dtype, np.uint8)


def as_packed_int32_array(data, dtype=None):
    """
    Interpret the input as PackedInt32Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_1d_array('PackedInt32Array', data, dtype, np.int32)


def as_packed_int64_array(data, dtype=None):
    """
    Interpret the input as PackedInt64Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_1d_array('PackedInt64Array', data, dtype, np.int64)


def as_packed_float32_array(data, dtype=None):
    """
    Interpret the input as PackedFloat32Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_1d_array('PackedFloat32Array', data, dtype, np.float32)


def as_packed_float64_array(data, dtype=None):
    """
    Interpret the input as PackedFloat32Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_1d_array('PackedFloat64Array', data, dtype, np.float64)


def as_packed_string_array(data, dtype=None):
    """
    Interpret the input as PackedStringArray.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_1d_array('PackedFloat64Array', data, dtype, np.dtypes.StringDType)


class _PackedArray1DBase(numpy.ndarray):
    def __array_finalize__(self, obj):
        ndim = self.ndim
        if ndim != 1:
            self.shape = (self.size,)


cdef class _PackedArray1DDataBase:
    cdef cvarray arr
    cdef npy_intp size

    def __init__(self):
        raise TypeError("Cannot init %r" % self.__class__)


cdef class _PackedByteArrayData(_PackedArray1DDataBase):
    cdef cpp.PackedByteArray _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedByteArray &p_arr):
        cdef _PackedByteArrayData self = _PackedByteArrayData.__new__(_PackedByteArrayData)
        self.size = p_arr.size()
        self.arr = <uint8_t[:self.size]>p_arr.ptrw()
        self._cpparr = p_arr


cdef class _PackedInt32ArrayData(_PackedArray1DDataBase):
    cdef cpp.PackedInt32Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedInt32Array &p_arr):
        cdef _PackedInt32ArrayData self = _PackedInt32ArrayData.__new__(_PackedInt32ArrayData)
        self.size = p_arr.size()
        self.arr = <int32_t[:self.size]>p_arr.ptrw()
        self._cpparr = p_arr

cdef class _PackedInt64ArrayData(_PackedArray1DDataBase):
    cdef cpp.PackedInt64Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedInt64Array &p_arr):
        cdef _PackedInt64ArrayData self = _PackedInt64ArrayData.__new__(_PackedInt64ArrayData)
        self.size = p_arr.size()
        self.arr = <int64_t[:self.size]>p_arr.ptrw()
        self._cpparr = p_arr

cdef class _PackedFloat32ArrayData(_PackedArray1DDataBase):
    cdef cpp.PackedFloat32Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedFloat32Array &p_arr):
        cdef _PackedFloat32ArrayData self = _PackedFloat32ArrayData.__new__(_PackedFloat32ArrayData)
        self.size = p_arr.size()
        self.arr = <float[:self.size]>p_arr.ptrw()
        self._cpparr = p_arr

cdef class _PackedFloat64ArrayData(_PackedArray1DDataBase):
    cdef cpp.PackedFloat64Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedFloat64Array &p_arr):
        cdef _PackedFloat64ArrayData self = _PackedFloat64ArrayData.__new__(_PackedFloat64ArrayData)
        self.size = p_arr.size()
        self.arr = <double[:self.size]>p_arr.ptrw()
        self._cpparr = p_arr


cdef inline numpy.ndarray array_from_packed_array_args(subtype, dtype, data, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)
    can_cast = kwargs.pop('can_cast', False)

    if isinstance(data, numpy.ndarray) and not copy:
        if data.dtype == dtype:
            base = data
        else:
            if not can_cast:
                cpp.UtilityFunctions.push_warning(
                    "Unexpected cast from %r to %r during %r initialization" % (data.dtype, dtype, subtype)
                )
            base = data.astype(dtype)
    else:
        base = np.array(data, dtype=dtype, copy=copy)

    if kwargs:
        raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

    return PyArraySubType_NewFromBase(subtype, base)


class PackedByteArray(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.uint8)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes" % subtype)
        try:
            tmp = dtype()
        except TypeError:
            tmp = dtype
        itemsize = tmp.itemsize
        if isinstance(itemsize, int) and itemsize != 1:
            raise TypeError("%r accepts only 1-byte datatypes" % subtype)

        return array_from_packed_array_args(subtype, dtype, data, kwargs)

    @classmethod
    def _from_cpp_data(subtype, _PackedByteArrayData base):
        cdef uint8_t [:] arr_view = base.arr

        # NOTE: No copying of array data, array buffer is reused
        return PyArraySubType_NewFromDataAndBase(subtype, arr_view, 1, NULL, <int>NPY_BYTE, base)


class PackedInt32Array(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.int32)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes" % subtype)
        try:
            tmp = dtype()
        except TypeError:
            tmp = dtype
        itemsize = tmp.itemsize
        if isinstance(itemsize, int) and itemsize != 2:
            raise TypeError("%r accepts only 2-byte datatypes" % subtype)

        return array_from_packed_array_args(subtype, dtype, data, kwargs)

    @classmethod
    def _from_cpp_data(subtype, _PackedInt32ArrayData base):
        cdef int32_t [:] arr_view = base.arr

        # NOTE: No copying of array data, array buffer is reused
        return PyArraySubType_NewFromDataAndBase(subtype, arr_view, 1, &base.size, <int>NPY_INT32, base)


class PackedInt64Array(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.int64)
        if not np.issubdtype(dtype, np.integer):
            raise TypeError("%r accepts only integer datatypes" % subtype)
        try:
            tmp = dtype()
        except TypeError:
            tmp = dtype
        itemsize = tmp.itemsize
        if isinstance(itemsize, int) and itemsize < 4:
            raise TypeError("%r accepts at least 4-byte datatypes" % subtype)

        return array_from_packed_array_args(subtype, dtype, data, kwargs)

    @classmethod
    def _from_cpp_data(subtype, _PackedInt64ArrayData base):
        cdef int64_t [:] arr_view = base.arr

        # NOTE: No copying of array data, array buffer is reused
        return PyArraySubType_NewFromDataAndBase(subtype, arr_view, 1, &base.size, <int>NPY_INT64, base)


class PackedFloat32Array(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)
        try:
            tmp = dtype()
        except TypeError:
            tmp = dtype
        itemsize = tmp.itemsize
        if isinstance(itemsize, int) and itemsize > 2:
            raise TypeError("%r accepts at most 2-byte datatypes" % subtype)

        return array_from_packed_array_args(subtype, dtype, data, kwargs)

    @classmethod
    def _from_cpp_data(subtype, _PackedFloat32ArrayData base):
        cdef float [:] arr_view = base.arr

        # NOTE: No copying of array data, array buffer is reused
        return PyArraySubType_NewFromDataAndBase(subtype, arr_view, 1, &base.size, <int>NPY_FLOAT32, base)


class PackedFloat64Array(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.float64)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("%r accepts only floating datatypes" % subtype)
        try:
            tmp = dtype()
        except TypeError:
            tmp = dtype
        itemsize = tmp.itemsize
        if isinstance(itemsize, int) and itemsize < 4:
            raise TypeError("%r accepts at least 4-byte datatypes" % subtype)

        return array_from_packed_array_args(subtype, dtype, data, kwargs)

    @classmethod
    def _from_cpp_data(subtype, _PackedFloat64ArrayData base):
        cdef double [:] arr_view = base.arr

        # NOTE: No copying of array data, array buffer is reused
        return PyArraySubType_NewFromDataAndBase(subtype, arr_view, 1, &base.size, <int>NPY_FLOAT64, base)


class PackedStringArray(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.dtypes.StringDType)
        if not isstring_dtype(dtype):
            raise TypeError("%r accepts only string-like datatypes" % subtype)

        return array_from_packed_array_args(subtype, dtype, data, kwargs)


cdef public object packed_byte_array_to_pyobject(cpp.PackedByteArray &p_arr):
    cdef _PackedByteArrayData data = _PackedByteArrayData.from_cpp(p_arr)

    return PackedByteArray._from_cpp_data(data)


cdef public object packed_int32_array_to_pyobject(cpp.PackedInt32Array &p_arr):
    cdef _PackedInt32ArrayData data = _PackedInt32ArrayData.from_cpp(p_arr)

    return PackedInt32Array._from_cpp_data(data)


cdef public object packed_int64_array_to_pyobject(cpp.PackedInt64Array &p_arr):
    cdef _PackedInt64ArrayData data = _PackedInt64ArrayData.from_cpp(p_arr)

    return PackedInt64Array._from_cpp_data(data)


cdef public object packed_float32_array_to_pyobject(cpp.PackedFloat32Array &p_arr):
    cdef _PackedFloat32ArrayData data = _PackedFloat32ArrayData.from_cpp(p_arr)

    return PackedFloat32Array._from_cpp_data(data)


cdef public object packed_float64_array_to_pyobject(cpp.PackedFloat64Array &p_arr):
    cdef _PackedFloat64ArrayData data = _PackedFloat64ArrayData.from_cpp(p_arr)

    return PackedFloat64Array._from_cpp_data(data)


cdef public object packed_string_array_to_pyobject(const cpp.PackedStringArray &p_arr):
    cdef int64_t size = p_arr.size(), i = 0

    cdef list pylist = PyList_New(size)
    cdef cpp.String item

    for i in range(size):
        pyitem = string_to_pyobject((p_arr.ptr() + i)[0])
        ref.Py_INCREF(pyitem)
        PyList_SET_ITEM(pylist, i, pyitem)

    return PackedStringArray(pylist)


cdef public object variant_packed_byte_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedByteArray arr = v.to_type[cpp.PackedByteArray]()

    return packed_byte_array_to_pyobject(arr)


cdef public object variant_packed_int32_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedInt32Array arr = v.to_type[cpp.PackedInt32Array]()

    return packed_int32_array_to_pyobject(arr)


cdef public object variant_packed_int64_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedInt64Array arr = v.to_type[cpp.PackedInt64Array]()

    return packed_int64_array_to_pyobject(arr)


cdef public object variant_packed_float32_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedFloat32Array arr = v.to_type[cpp.PackedFloat32Array]()

    return packed_float32_array_to_pyobject(arr)


cdef public object variant_packed_float64_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedFloat64Array arr = v.to_type[cpp.PackedFloat64Array]()

    return packed_float64_array_to_pyobject(arr)


cdef public object variant_packed_string_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedStringArray arr = v.to_type[cpp.PackedStringArray]()

    return packed_string_array_to_pyobject(arr)


cdef public void packed_string_array_from_pyobject(object p_obj, cpp.PackedStringArray *r_ret) noexcept:
    cdef cpp.PackedStringArray arr = cpp.PackedStringArray()
    cdef int64_t size, i
    cdef object pyitem

    if PySequence_Check(p_obj):
        size = PySequence_Size(p_obj)
        arr.resize(size)

        for i in range(size):
            pyitem = PySequence_GetItem(p_obj, i)
            string_from_pyobject(pyitem, arr.ptrw() + i)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedStringArray" % p_obj)

    r_ret[0] = arr


cdef public void variant_packed_string_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedStringArray arr
    packed_string_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)
