cdef inline object _as_any_packed_array(type_name, data, dtype, default_dtype, size_t lastdimsize=0, as_color=False):
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
        tmp = dtype()
        itemsize = tmp.itemsize
        copy = True

    if lastdimsize == 0:
        if np.issubdtype(dtype, np.floating):
            if itemsize > 4 or itemsize == -1:
                return PackedFloat64Array(data, dtype=dtype, copy=copy)
            else:
                return PackedFloat32Array(data, dtype=dtype, copy=copy)
        elif isstring_dtype(dtype):
            return PackedStringArray(data, dtype=dtype, copy=copy)
        elif not np.issubdtype(dtype, np.integer):
            raise ValueError("Packed<Type>Array data must be numeric or string-like, "
                             "requested dtype (%s) is not recognized as such." % dtype)
        if itemsize == 1:
            return PackedByteArray(data, dtype=dtype, copy=copy)
        else:
            if itemsize > 4 or itemsize == -1:
                return PackedInt64Array(data, dtype=dtype, copy=copy)
            else:
                return PackedInt32Array(data, dtype=dtype, copy=copy)
    elif lastdimsize == 2:
        return PackedVector2Array(data, dtype=dtype, copy=copy)
    elif lastdimsize == 3:
        return PackedVector3Array(data, dtype=dtype, copy=copy)
    elif lastdimsize == 4:
        if as_color:
            return PackedColorArray(data, dtype=dtype, copy=copy)
        else:
            return PackedVector4Array(data, dtype=dtype, copy=copy)


def as_packed_byte_array(data, dtype=None):
    """
    Interpret the input as PackedByteArray.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_packed_array('PackedByteArray', data, dtype, np.uint8)


def as_packed_int32_array(data, dtype=None):
    """
    Interpret the input as PackedInt32Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_packed_array('PackedInt32Array', data, dtype, np.int32)


def as_packed_int64_array(data, dtype=None):
    """
    Interpret the input as PackedInt64Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_packed_array('PackedInt64Array', data, dtype, np.int64)


def as_packed_float32_array(data, dtype=None):
    """
    Interpret the input as PackedFloat32Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_packed_array('PackedFloat32Array', data, dtype, np.float32)


def as_packed_float64_array(data, dtype=None):
    """
    Interpret the input as PackedFloat32Array.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_packed_array('PackedFloat64Array', data, dtype, np.float64)


def as_packed_string_array(data, dtype=None):
    """
    Interpret the input as PackedStringArray.

    If 'dtype' argument is passed interpret the result as
    one of PackedByteArray, PackedInt32Array, PackedInt64Array,
    PackedFloat32Array, PackedFloat64Array or PackedStringArray base on
    what array type is closer to the requested dtype. Multi-dimensional
    arrays will be flattened.
    """
    return _as_any_packed_array('PackedFloat64Array', data, dtype, np.dtypes.StringDType)


def as_packed_vector2_array(data, dtype=None):
    """
    Interpret the input as PackedVector2Array.
    """
    return _as_any_packed_array('PackedVector2Array', data, dtype, np.float32, 2)


def as_packed_vector3_array(data, dtype=None):
    """
    Interpret the input as PackedVector3Array.
    """
    return _as_any_packed_array('PackedVector3rray', data, dtype, np.float32, 3)


def as_packed_color_array(data, dtype=None):
    """
    Interpret the input as PackedColorArray.
    """
    return _as_any_packed_array('PackedColorArray', data, dtype, np.float32, 4, True)


def as_packed_vector4_array(data, dtype=None):
    """
    Interpret the input as PackedVector2Array.
    """
    return _as_any_packed_array('PackedVector2Array', data, dtype, np.float32, 4)


class _PackedArray1DBase(numpy.ndarray):
    def __array_finalize__(self, obj):
        if isinstance(obj, _PackedArray1DBase):
            return

        ndim = self.ndim
        if ndim != 1:
            self.shape = (self.size,)


class _PackedArray2DBase(numpy.ndarray):
    def __array_finalize__(self, obj):
        if isinstance(obj, _PackedArray2DBase):
            return

        ndim = self.ndim
        lastdimsize = self._lastdimsize
        if ndim != 2:
            self.shape = (self.size // lastdimsize, lastdimsize)


cdef class _PackedByteArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedByteArray _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedByteArray &p_arr):
        cdef _PackedByteArrayData self = _PackedByteArrayData.__new__(_PackedByteArrayData)
        self._cpparr = p_arr
        cdef npy_intp size = p_arr.size()
        self.arr = PyArray_New(PackedByteArray, 1, &size, NPY_UINT8, NULL, self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedInt32ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedInt32Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedInt32Array &p_arr):
        cdef _PackedInt32ArrayData self = _PackedInt32ArrayData.__new__(_PackedInt32ArrayData)
        self._cpparr = p_arr
        cdef npy_intp size = p_arr.size()
        self.arr = PyArray_New(PackedInt32Array, 1, &size, NPY_INT32, NULL, self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedInt64ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedInt64Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedInt64Array &p_arr):
        cdef _PackedInt64ArrayData self = _PackedInt64ArrayData.__new__(_PackedInt64ArrayData)
        self._cpparr = p_arr
        cdef npy_intp size = p_arr.size()
        self.arr = PyArray_New(PackedInt64Array, 1, &size, NPY_INT64, NULL, self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)
        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedFloat32ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedFloat32Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedFloat32Array &p_arr):
        cdef _PackedFloat32ArrayData self = _PackedFloat32ArrayData.__new__(_PackedFloat32ArrayData)
        self._cpparr = p_arr
        cdef npy_intp size = p_arr.size()
        self.arr = PyArray_New(PackedFloat32Array, 1, &size, NPY_FLOAT32, NULL, self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedFloat64ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedFloat64Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedFloat64Array &p_arr):
        cdef _PackedFloat64ArrayData self = _PackedFloat64ArrayData.__new__(_PackedFloat64ArrayData)
        self._cpparr = p_arr
        cdef npy_intp size = p_arr.size()
        self.arr = PyArray_New(PackedFloat64Array, 1, &size, NPY_FLOAT64, NULL, self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedVector2ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedVector2Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedVector2Array &p_arr):
        cdef _PackedVector2ArrayData self = _PackedVector2ArrayData.__new__(_PackedVector2ArrayData)
        self._cpparr = p_arr
        cdef npy_intp[2] shape = [p_arr.size() // 2, 2]
        self.arr = PyArray_New(PackedVector2Array, 2, shape, NPY_FLOAT32, NULL, <float *>self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedVector3ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedVector3Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedVector3Array &p_arr):
        cdef _PackedVector3ArrayData self = _PackedVector3ArrayData.__new__(_PackedVector3ArrayData)
        self._cpparr = p_arr
        cdef npy_intp[2] shape = [p_arr.size() // 3, 3]
        self.arr = PyArray_New(PackedVector3Array, 2, shape, NPY_FLOAT32, NULL, <float *>self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedColorArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedColorArray _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedColorArray &p_arr):
        cdef _PackedColorArrayData self = _PackedColorArrayData.__new__(_PackedColorArrayData)
        self._cpparr = p_arr
        cdef npy_intp[2] shape = [p_arr.size() // 4, 4]
        self.arr = PyArray_New(PackedColorArray, 2, shape, NPY_FLOAT32, NULL, <float *>self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef class _PackedVector4ArrayData:
    cdef numpy.ndarray arr
    cdef cpp.PackedVector4Array _cpparr

    @staticmethod
    cdef from_cpp(cpp.PackedVector4Array &p_arr):
        cdef _PackedVector4ArrayData self = _PackedVector4ArrayData.__new__(_PackedVector4ArrayData)
        self._cpparr = p_arr
        cdef npy_intp[2] shape = [p_arr.size() // 4, 4]
        self.arr = PyArray_New(PackedVector4Array, 2, shape, NPY_FLOAT32, NULL, <float *>self._cpparr.ptrw(), 0,
                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, None)

        ref.Py_INCREF(self)
        numpy.PyArray_SetBaseObject(self.arr, self)

        return self


cdef inline numpy.ndarray array_from_packed_array1d_args(subtype, dtype, data, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)

    if isinstance(data, numpy.ndarray) and not copy:
        if data.dtype == dtype:
            base = data
        else:
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

        return array_from_packed_array1d_args(subtype, dtype, data, kwargs)


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
        if isinstance(itemsize, int) and itemsize > 4:
            raise TypeError("%r accepts at most 4-byte datatypes" % subtype)

        return array_from_packed_array1d_args(subtype, dtype, data, kwargs)


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
        if isinstance(itemsize, int) and itemsize < 8:
            raise TypeError("%r accepts at least 8-byte datatypes" % subtype)

        return array_from_packed_array1d_args(subtype, dtype, data, kwargs)


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
        if isinstance(itemsize, int) and itemsize > 4:
            raise TypeError("%r accepts at most 4-byte datatypes" % subtype)

        return array_from_packed_array1d_args(subtype, dtype, data, kwargs)


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
        if isinstance(itemsize, int) and itemsize < 8:
            raise TypeError("%r accepts at least 8-byte datatypes" % subtype)

        return array_from_packed_array1d_args(subtype, dtype, data, kwargs)


class PackedStringArray(_PackedArray1DBase):
    def __new__(subtype, data, **kwargs):
        dtype = kwargs.pop('dtype', np.dtypes.StringDType)
        if not isstring_dtype(dtype):
            raise TypeError("%r accepts only string-like datatypes" % subtype)

        return array_from_packed_array1d_args(subtype, dtype, data, kwargs)


cdef inline numpy.ndarray array_from_packed_array2d_args(subtype, data, kwargs):
    cdef numpy.ndarray base

    copy = kwargs.pop('copy', True)
    dtype = kwargs.pop('dtype', np.float32)

    if not np.issubdtype(dtype, np.number):
        raise TypeError("%r accepts only numeric datatypes" % subtype)

    if isinstance(data, numpy.ndarray) and not copy:
        shape = data.shape
        if data.dtype == dtype and len(shape) == 2 and shape[1] == subtype._lastdimsize:
            base = data
        else:
            cpp.UtilityFunctions.push_warning(
                "Unexpected cast from %r to %r during %r initialization" % (data.dtype, dtype, subtype)
            )
            base = data.astype(dtype)
    else:
        base = np.array(data, dtype=dtype, copy=copy)

    if kwargs:
        raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

    return PyArraySubType_NewFromBase(subtype, base)


class PackedVector2Array(_PackedArray2DBase):
    _lastdimsize = 2

    def __new__(subtype, data, **kwargs):
        return array_from_packed_array2d_args(subtype, data, kwargs)

    def __getitem__(self, index):
        cdef object obj = super().__getitem__(index)

        if isinstance(obj, PackedVector2Array):
            return as_vector2(obj)
        else:
            return obj


class PackedVector3Array(_PackedArray2DBase):
    _lastdimsize = 3

    def __new__(subtype, data, **kwargs):
        return array_from_packed_array2d_args(subtype, data, kwargs)

    def __getitem__(self, index):
        cdef object obj = super().__getitem__(index)

        if isinstance(obj, PackedVector3Array):
            return as_vector3(obj)
        else:
            return obj


class PackedColorArray(_PackedArray2DBase):
    _lastdimsize = 4

    def __new__(subtype, data, **kwargs):
        if not isinstance(data, numpy.ndarray) and issubscriptable(data):
            for i, item in enumerate(data):
                if not issubscriptable(item) or len(item) > 4 or len(item) < 3:
                    raise ValueError("Incompatible %r data" % subtype)
                if len(item) == 3:
                    data[i] = (*item, 1.)

        return array_from_packed_array2d_args(subtype, data, kwargs)

    def __getitem__(self, index):
        cdef object obj = super().__getitem__(index)

        if isinstance(obj, PackedColorArray):
            return as_color(obj)
        else:
            return obj


class PackedVector4Array(_PackedArray2DBase):
    _lastdimsize = 4

    def __new__(subtype, data, **kwargs):
        return array_from_packed_array2d_args(subtype, data, kwargs)

    def __getitem__(self, index):
        cdef object obj = super().__getitem__(index)

        if isinstance(obj, PackedVector4Array):
            return as_vector4(obj)
        else:
            return obj


cdef public object packed_byte_array_to_pyobject(const cpp.PackedByteArray &p_arr):
    cdef _PackedByteArrayData data = _PackedByteArrayData.from_cpp(<cpp.PackedByteArray &>p_arr)

    return data.arr


cdef public object packed_int32_array_to_pyobject(const cpp.PackedInt32Array &p_arr):
    cdef _PackedInt32ArrayData data = _PackedInt32ArrayData.from_cpp(<cpp.PackedInt32Array &>p_arr)

    return data.arr


cdef public object packed_int64_array_to_pyobject(const cpp.PackedInt64Array &p_arr):
    cdef _PackedInt64ArrayData data = _PackedInt64ArrayData.from_cpp(<cpp.PackedInt64Array &>p_arr)

    return data.arr


cdef public object packed_float32_array_to_pyobject(const cpp.PackedFloat32Array &p_arr):
    cdef _PackedFloat32ArrayData data = _PackedFloat32ArrayData.from_cpp(<cpp.PackedFloat32Array &>p_arr)

    return data.arr


cdef public object packed_float64_array_to_pyobject(const cpp.PackedFloat64Array &p_arr):
    cdef _PackedFloat64ArrayData data = _PackedFloat64ArrayData.from_cpp(<cpp.PackedFloat64Array &>p_arr)

    return data.arr


# NOTE: PackedStringArrays are converted to lists by default
cdef public object packed_string_array_to_pyobject(const cpp.PackedStringArray &p_arr):
    cdef int64_t size = p_arr.size(), i = 0

    cdef list pylist = PyList_New(size)
    cdef cpp.String item

    for i in range(size):
        pyitem = string_to_pyobject((p_arr.ptr() + i)[0])
        ref.Py_INCREF(pyitem)
        PyList_SET_ITEM(pylist, i, pyitem)

    return pylist


cdef public object packed_vector2_array_to_pyobject(const cpp.PackedVector2Array &p_arr):
    cdef _PackedVector2ArrayData data = _PackedVector2ArrayData.from_cpp(<cpp.PackedVector2Array &>p_arr)

    return data.arr


cdef public object packed_vector3_array_to_pyobject(const cpp.PackedVector3Array &p_arr):
    cdef _PackedVector3ArrayData data = _PackedVector3ArrayData.from_cpp(<cpp.PackedVector3Array &>p_arr)

    return data.arr


cdef public object packed_color_array_to_pyobject(const cpp.PackedColorArray &p_arr):
    cdef _PackedColorArrayData data = _PackedColorArrayData.from_cpp(<cpp.PackedColorArray &>p_arr)

    return data.arr


cdef public object packed_vector4_array_to_pyobject(const cpp.PackedVector4Array &p_arr):
    cdef _PackedVector4ArrayData data = _PackedVector4ArrayData.from_cpp(<cpp.PackedVector4Array &>p_arr)

    return data.arr


cdef public object variant_packed_byte_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedByteArray arr = v.to_type[cpp.PackedByteArray]()
    cdef _PackedByteArrayData data = _PackedByteArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_int32_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedInt32Array arr = v.to_type[cpp.PackedInt32Array]()
    cdef _PackedInt32ArrayData data = _PackedInt32ArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_int64_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedInt64Array arr = v.to_type[cpp.PackedInt64Array]()
    cdef _PackedInt64ArrayData data = _PackedInt64ArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_float32_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedFloat32Array arr = v.to_type[cpp.PackedFloat32Array]()
    cdef _PackedFloat32ArrayData data = _PackedFloat32ArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_float64_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedFloat64Array arr = v.to_type[cpp.PackedFloat64Array]()
    cdef _PackedFloat64ArrayData data = _PackedFloat64ArrayData.from_cpp(arr)

    return data.arr


# NOTE: PackedStringArrays are converted to lists by default
cdef public object variant_packed_string_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedStringArray arr = v.to_type[cpp.PackedStringArray]()
    cdef int64_t size = arr.size(), i = 0

    cdef list pylist = PyList_New(size)
    cdef cpp.String item

    for i in range(size):
        pyitem = string_to_pyobject((arr.ptr() + i)[0])
        ref.Py_INCREF(pyitem)
        PyList_SET_ITEM(pylist, i, pyitem)

    return pylist


cdef public object variant_packed_vector2_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedVector2Array arr = v.to_type[cpp.PackedVector2Array]()
    cdef _PackedVector2ArrayData data = _PackedVector2ArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_vector3_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedVector3Array arr = v.to_type[cpp.PackedVector3Array]()
    cdef _PackedVector3ArrayData data = _PackedVector3ArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_color_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedColorArray arr = v.to_type[cpp.PackedColorArray]()
    cdef _PackedColorArrayData data = _PackedColorArrayData.from_cpp(arr)

    return data.arr


cdef public object variant_packed_vector4_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedVector4Array arr = v.to_type[cpp.PackedVector4Array]()
    cdef _PackedVector4ArrayData data = _PackedVector4ArrayData.from_cpp(arr)

    return data.arr


ctypedef fused array_data_t:
    uint8_t
    int32_t
    int64_t
    float
    double


cdef inline _copy_array_data(array_data_t *ptrw, numpy.ndarray arr):
    cdef int64_t size = arr.size, i = 0
    cdef array_data_t [:] cview = <array_data_t [:size]>ptrw
    cdef array_data_t [:] pyview = arr

    cview[:] = pyview


cdef inline _copy_array2d_data(array_data_t *ptrw, numpy.ndarray arr):
    cdef int64_t size1 = arr.shape[0], size2 = arr.shape[1], i = 0
    cdef array_data_t [:, :] cview = <array_data_t [:size1, :size2]>ptrw
    cdef array_data_t [:, :] pyview = arr

    cview[...] = pyview


cdef public void packed_byte_array_from_pyobject(object p_obj, cpp.PackedByteArray *r_ret) noexcept:
    if not isinstance(p_obj, PackedByteArray) or not p_obj.dtype == np.uint8:
        p_obj = as_packed_byte_array(p_obj, dtype=np.uint8)

    cdef object _base = p_obj.base
    cdef _PackedByteArrayData base
    cdef cpp.PackedByteArray ret
    cdef int64_t size

    if isinstance(_base, _PackedByteArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array_data[uint8_t](ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedByteArray")

    r_ret[0] = ret


cdef public void packed_int32_array_from_pyobject(object p_obj, cpp.PackedInt32Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedInt32Array) or not p_obj.dtype == np.int32:
        p_obj = as_packed_int32_array(p_obj, dtype=np.int32)

    cdef object _base = p_obj.base
    cdef _PackedInt32ArrayData base
    cdef cpp.PackedInt32Array ret
    cdef int64_t size

    if isinstance(_base, _PackedInt32ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array_data[int32_t](ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void packed_int64_array_from_pyobject(object p_obj, cpp.PackedInt64Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedInt64Array) or not p_obj.dtype == np.int64:
        p_obj = as_packed_int32_array(p_obj, dtype=np.int64)

    cdef object _base = p_obj.base
    cdef _PackedInt64ArrayData base
    cdef cpp.PackedInt64Array ret
    cdef int64_t size

    if isinstance(_base, _PackedInt64ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array_data[int64_t](ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void packed_float32_array_from_pyobject(object p_obj, cpp.PackedFloat32Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedFloat32Array) or not p_obj.dtype == np.float32:
        p_obj = as_packed_float32_array(p_obj, dtype=np.float32)

    cdef object _base = p_obj.base
    cdef _PackedFloat32ArrayData base
    cdef cpp.PackedFloat32Array ret
    cdef int64_t size

    if isinstance(_base, _PackedFloat32ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array_data[float](ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void packed_float64_array_from_pyobject(object p_obj, cpp.PackedFloat64Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedFloat64Array) or not p_obj.dtype == np.float64:
        p_obj = as_packed_float64_array(p_obj, dtype=np.float64)

    cdef object _base = p_obj.base
    cdef _PackedFloat64ArrayData base
    cdef cpp.PackedFloat64Array ret
    cdef int64_t size

    if isinstance(_base, _PackedFloat64ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array_data[double](ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


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


cdef public void packed_vector2_array_from_pyobject(object p_obj, cpp.PackedVector2Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedVector2Array) or not p_obj.dtype == np.float32:
        p_obj = as_packed_vector2_array(p_obj, dtype=np.float32)

    cdef object _base = p_obj.base
    cdef _PackedVector2ArrayData base
    cdef cpp.PackedVector2Array ret
    cdef int64_t size

    if isinstance(_base, _PackedVector2ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array2d_data[float](<float *>ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void packed_vector3_array_from_pyobject(object p_obj, cpp.PackedVector3Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedVector3Array) or not p_obj.dtype == np.float32:
        p_obj = as_packed_vector3_array(p_obj, dtype=np.float32)

    cdef object _base = p_obj.base
    cdef _PackedVector3ArrayData base
    cdef cpp.PackedVector3Array ret
    cdef int64_t size

    if isinstance(_base, _PackedVector3ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array2d_data[float](<float *>ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void packed_color_array_from_pyobject(object p_obj, cpp.PackedColorArray *r_ret) noexcept:
    if not isinstance(p_obj, PackedColorArray) or not p_obj.dtype == np.float32:
        p_obj = as_packed_color_array(p_obj, dtype=np.float32)

    cdef object _base = p_obj.base
    cdef _PackedColorArrayData base
    cdef cpp.PackedColorArray ret
    cdef int64_t size

    if isinstance(_base, _PackedVector3ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array2d_data[float](<float *>ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void packed_vector4_array_from_pyobject(object p_obj, cpp.PackedVector4Array *r_ret) noexcept:
    if not isinstance(p_obj, PackedVector4Array) or not p_obj.dtype == np.float32:
        p_obj = as_packed_vector4_array(p_obj, dtype=np.float32)

    cdef object _base = p_obj.base
    cdef _PackedVector4ArrayData base
    cdef cpp.PackedVector4Array ret
    cdef int64_t size

    if isinstance(_base, _PackedVector4ArrayData):
        base = _base
        ret = base._cpparr
    elif isinstance(_base, numpy.ndarray):
        size = p_obj.size
        ret.resize(size)
        _copy_array2d_data[float](<float *>ret.ptrw(), p_obj)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedInt32Array")

    r_ret[0] = ret


cdef public void variant_packed_byte_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedByteArray arr
    packed_byte_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_int32_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedInt32Array arr
    packed_int32_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_int64_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedInt64Array arr
    packed_int64_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_float32_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedFloat32Array arr
    packed_float32_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_float64_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedFloat64Array arr
    packed_float64_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_string_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedStringArray arr
    packed_string_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_vector2_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedVector2Array arr
    packed_vector2_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_vector3_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedVector3Array arr
    packed_vector3_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_color_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedColorArray arr
    packed_color_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)


cdef public void variant_packed_vector4_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedVector4Array arr
    packed_vector4_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)
