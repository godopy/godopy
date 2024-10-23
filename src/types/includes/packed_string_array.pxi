cpdef numpy.ndarray as_packed_string_array(data, dtype=None):
    """
    Interpret the input as PackedStringArray
    """
    if dtype is None:
        dtype = np.dtypes.StringDType

    if not isstring_dtype(dtype):
        raise ValueError("Unsupported dtype for PackedStringArray")

    copy = False
    if isinstance(data, numpy.ndarray):
        if data.dtype != dtype:
            copy = True
    else:
        copy = True

    return PackedStringArray(data, dtype=dtype, copy=copy, can_cast=True)


class PackedStringArray(numpy.ndarray):
    def __new__(subtype, data, **kwargs):
        cdef numpy.ndarray base

        dtype = kwargs.pop('dtype', np.dtypes.StringDType)
        if not isstring_dtype(dtype):
            raise ValueError("Unsupported dtype for %r" % subtype)
        
        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)

        if kwargs:
            raise TypeError("Invalid keyword argument %r" % list(kwargs.keys()).pop())

        if issubscriptable(data):
            if isinstance(data, numpy.ndarray) and not copy:
                if data.dtype == dtype:
                    base = data
                else:
                    if not can_cast:
                        cpp.UtilityFunctions.push_warning(
                            "Unexcpected cast from %r to %r during %r initialization" % (data.dtype, dtype, subtype)
                        )
                    base = data.astype(dtype)
            else:
                base = np.array(data, dtype=dtype, copy=copy)
        else:
            raise ValueError("Unsupported data %r for %r" % (type(data), subtype))

        cdef numpy.ndarray ret = PyArraySubType_NewFromBase(subtype, base)

        return ret


cdef public object packed_string_array_to_pyobject(const cpp.PackedStringArray &p_arr):
    cdef int64_t size = p_arr.size(), i = 0

    cdef list pylist = PyList_New(size)
    cdef cpp.String item
    # cdef String pyitem

    for i in range(size):
        # item = (p_arr.ptr() + i)[0]
        pyitem = string_to_pyobject((p_arr.ptr() + i)[0])
        ref.Py_INCREF(pyitem)
        PyList_SET_ITEM(pylist, i, pyitem)

    return PackedStringArray(pylist)


cdef public object variant_packed_string_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.PackedStringArray arr = v.to_type[cpp.PackedStringArray]()

    return packed_string_array_to_pyobject(arr)


cdef public void packed_string_array_from_pyobject(object p_obj, cpp.PackedStringArray *r_ret) noexcept:
    cdef cpp.PackedStringArray arr = cpp.PackedStringArray()
    cdef int64_t size, i
    cdef object pyitem
    # cdef cpp.String item

    if PySequence_Check(p_obj):
        size = PySequence_Size(p_obj)
        arr.resize(size)

        for i in range(size):
            pyitem = PySequence_GetItem(p_obj, i)
            string_from_pyobject(pyitem, arr.ptrw() + i)
            # (arr.ptrw() + i)[0] = item
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ PackedStringArray" % p_obj)

    r_ret[0] = arr


cdef public void variant_packed_string_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.PackedStringArray arr
    packed_string_array_from_pyobject(p_obj, &arr)

    r_ret[0] = cpp.Variant(arr)
