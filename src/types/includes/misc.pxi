def as_string_name(other):
    return StringName(other)


cdef class StringName(str):
    def __init__(self, other=None):
        # NOTE: other will initialize itself automatically in the str.__new__

        super().__init__()

        sys.intern(str(self))
        string_name_from_pyobject(self, &self._base)

    cdef void *ptr(self):
        return self._base._native_ptr()

    begins_with = str.startswith

    def bin_to_int(self):
        return int(self, 2)

    # TODO: All documented methods


cdef public object string_name_to_pyobject(const cpp.StringName &p_val):
    cdef cpp.String s = cpp.String(p_val)
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(s._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(s._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return StringName(PyUnicode_FromWideChar(wstr.get_data(), len))


cdef public object variant_string_name_to_pyobject(const cpp.Variant &v):
    cdef cpp.String ret = v.to_type[cpp.String]()
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(ret._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(ret._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return StringName(PyUnicode_FromWideChar(wstr.get_data(), len))


cdef public void string_name_from_pyobject(object p_obj, cpp.StringName *r_ret) noexcept:
    cdef const wchar_t *wstr
    cdef const char *cstr
    cdef bytes tmp
    cdef cpp.String s

    if PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars(&s, wstr)
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        gdextension_interface_string_new_with_utf8_chars(&s, cstr)
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        gdextension_interface_string_new_with_utf8_chars(&s, cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ StringName" % p_obj)
        s = cpp.String()

    r_ret[0] = cpp.StringName(s)


cdef public void variant_string_name_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.StringName ret
    string_name_from_pyobject(p_obj, &ret)

    r_ret[0] = cpp.Variant(ret)


def as_node_path(other):
    return NodePath(other)


class NodePath(pathlib.PurePosixPath):
    def is_empty(self):
        return not self.parts

    # TODO: All documented methods


cdef public object node_path_to_pyobject(const cpp.NodePath &p_val):
    cdef cpp.String s = cpp.String(p_val)
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(s._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(s._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return NodePath(PyUnicode_FromWideChar(wstr.get_data(), len))


cdef public object variant_node_path_to_pyobject(const cpp.Variant &v):
    cdef cpp.String ret = v.to_type[cpp.String]()
    cdef cpp.CharWideString wstr
    cdef int64_t len = gdextension_interface_string_to_wide_chars(ret._native_ptr(), NULL, 0)
    wstr.resize(len + 1)
    gdextension_interface_string_to_wide_chars(ret._native_ptr(), wstr.ptrw(), len)
    wstr.set_zero(len)

    return NodePath(PyUnicode_FromWideChar(wstr.get_data(), len))


cdef public void node_path_from_pyobject(object p_obj, cpp.NodePath *r_ret) noexcept:
    cdef const wchar_t *wstr
    cdef const char *cstr
    cdef bytes tmp
    cdef cpp.String s

    if isinstance(p_obj, pathlib.PurePosixPath):
        wstr = PyUnicode_AsWideCharString(str(p_obj), NULL)
        gdextension_interface_string_new_with_wide_chars(&s, wstr)
    elif PyUnicode_Check(p_obj):
        wstr = PyUnicode_AsWideCharString(p_obj, NULL)
        gdextension_interface_string_new_with_wide_chars(&s, wstr)
    elif PyBytes_Check(p_obj):
        cstr = PyBytes_AsString(p_obj)
        gdextension_interface_string_new_with_utf8_chars(&s, cstr)
    elif isinstance(p_obj, np.flexible):
        tmp = bytes(p_obj)
        cstr = PyBytes_AsString(tmp)
        gdextension_interface_string_new_with_utf8_chars(&s, cstr)
    else:
        cpp.UtilityFunctions.push_error("Could not convert %r to C++ NodePath" % p_obj)
        s = cpp.String()

    r_ret[0] = cpp.NodePath(s)


cdef public void variant_node_path_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.NodePath ret
    node_path_from_pyobject(p_obj, &ret)

    r_ret[0] = cpp.Variant(ret)


cdef class RID:
    @staticmethod
    cdef RID from_cpp_rid(const cpp._RID &p_val):
        
        cdef RID self = RID.__new__(RID)
        self._base = p_val

        return self

    def __init__(self, object arg=None):
        if isinstance(arg, RID):
            self._base = cpp._RID((<RID>arg)._base)
        elif arg is not None:
            raise ValueError("Only 'RID' or no arguments are allowed, got %r" % type(arg))
        else:
            self._base = cpp._RID()

    def __int__(self):
        return self._base.get_id()

    def __repr__(self):
        return "%s.%s(%d)" % (self.__class__.__module__, self.__class__.__name__, self)

    def get_id(self):
        return int(self)

    def is_valid(self):
        return self._base.is_valid()


cdef public object rid_to_pyobject(const cpp._RID &p_val):
    return RID.from_cpp_rid(p_val)


cdef public object variant_rid_to_pyobject(const cpp.Variant &v):
    cdef cpp._RID rid = v.to_type[cpp._RID]()

    return RID.from_cpp_rid(rid)


cdef public void rid_from_pyobject(object p_obj, cpp._RID *r_ret) noexcept:
    if isinstance(p_obj, RID):
        r_ret[0] = (<RID>p_obj)._base
    else:
        cpp.UtilityFunctions.push_error("'RID' is required, got %r" % type(p_obj))

        r_ret[0] = cpp._RID()


cdef public void variant_rid_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp._RID ret

    if isinstance(p_obj, RID):
        ret = (<RID>p_obj)._base
    else:
        cpp.UtilityFunctions.push_error("'RID' is required, got %r" % type(p_obj))
        ret = cpp._RID()

    r_ret[0] = cpp.Variant(ret)


cdef class Callable:
    @staticmethod
    cdef Callable from_cpp(const cpp.GodotCppCallable &p_val):
        cdef Callable self = Callable.__new__(Callable)
        self._base = p_val

        return self

    # TODO: Custom Callables

    def __init__(self, object arg=None):
        if isinstance(arg, Callable):
            self._base = cpp.GodotCppCallable((<Callable>arg)._base)
        elif arg is not None:
            raise ValueError("Invalid positional argument 1, a 'Callable' is required, got %r" % type(arg))
        else:
            self._base = cpp.GodotCppCallable()


cdef public object callable_to_pyobject(const cpp.GodotCppCallable &c):
    return Callable.from_cpp(c)


cdef public object variant_callable_to_pyobject(const cpp.Variant &v):
    cdef cpp.GodotCppCallable c = v.to_type[cpp.GodotCppCallable]()

    return Callable.from_cpp(c)


cdef public void callable_from_pyobject(object p_obj, cpp.GodotCppCallable *r_ret) noexcept:
    if isinstance(p_obj, Callable):
        r_ret[0] = (<Callable>p_obj)._base
    else:
        cpp.UtilityFunctions.push_error("'Callable' is required, got %r" % type(p_obj))

        r_ret[0] = cpp.GodotCppCallable()


cdef public void variant_callable_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.GodotCppCallable ret

    if isinstance(p_obj, Callable):
        ret = (<Callable>p_obj)._base
    else:
        cpp.UtilityFunctions.push_error("'Callable' is required, got %r" % type(p_obj))
        ret = cpp.GodotCppCallable()

    r_ret[0] = cpp.Variant(ret)


cdef class Signal:
    @staticmethod
    cdef Signal from_cpp(const cpp.GodotCppSignal &p_val):
        cdef Signal self = Signal.__new__(Signal)
        self._base = p_val

        return self

    # TODO: Custom Signals

    def __init__(self, object arg=None):
        if isinstance(arg, Signal):
            self._base = cpp.GodotCppSignal((<Signal>arg)._base)
        elif arg is not None:
            raise ValueError("Invalid positional argument 1, a 'Signal' is required, got %r" % type(arg))
        else:
            self._base = cpp.GodotCppSignal()


cdef public object signal_to_pyobject(const cpp.GodotCppSignal &s):
    return Signal.from_cpp(s)


cdef public object variant_signal_to_pyobject(const cpp.Variant &v):
    cdef cpp.GodotCppSignal s = v.to_type[cpp.GodotCppSignal]()

    return Signal.from_cpp(s)


cdef public void signal_from_pyobject(object p_obj, cpp.GodotCppSignal *r_ret) noexcept:
    if isinstance(p_obj, Signal):
        r_ret[0] = (<Signal>p_obj)._base
    else:
        cpp.UtilityFunctions.push_error("'Signal' is required, got %r" % type(p_obj))

        r_ret[0] = cpp.GodotCppSignal()


cdef public void variant_signal_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.GodotCppSignal ret

    if isinstance(p_obj, Signal):
        ret = (<Signal>p_obj)._base
    else:
        cpp.UtilityFunctions.push_error("'Signal' is required, got %r" % type(p_obj))
        ret = cpp.GodotCppSignal()

    r_ret[0] = cpp.Variant(ret)


Dictionary = dict


cdef public object dictionary_to_pyobject(const cpp.Dictionary &p_val):
    cdef cpp.Array keys = p_val.keys()
    cdef int64_t size = keys.size(), i = 0

    cdef ret = {}
    cdef cpp.Variant key
    cdef cpp.Variant value
    cdef object pykey
    cdef object pyvalue

    for i in range(size):
        key = cpp.godot_array_get_item(keys, i)
        pykey = variant_to_pyobject(key)
        value = cpp.godot_dictionary_get_item(p_val, key)
        pyvalue = variant_to_pyobject(value)

        ret[pykey] = pyvalue

    return ret


cdef public object variant_dictionary_to_pyobject(const cpp.Variant &v):
    cdef cpp.Dictionary ret = v.to_type[cpp.Dictionary]()

    return dictionary_to_pyobject(ret)


cdef public void dictionary_from_pyobject(object p_obj, cpp.Dictionary *r_ret) noexcept:
    cdef cpp.Dictionary ret

    cdef cpp.Variant key
    cdef cpp.Variant value

    if PyMapping_Check(p_obj):
        for pykey, pyvalue in p_obj.items():
            variant_from_pyobject(pykey, &key)
            variant_from_pyobject(pyvalue, &value)
            cpp.godot_dictionary_set_item(ret, key, value)
    else:
        cpp.UtilityFunctions.push_error("a mapping is required, got %r" % type(p_obj))

    r_ret[0] = ret.duplicate()


cdef public void variant_dictionary_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef cpp.Dictionary ret

    cdef cpp.Variant key
    cdef cpp.Variant value

    if PyMapping_Check(p_obj):
        for pykey, pyvalue in p_obj.items():
            variant_from_pyobject(pykey, &key)
            variant_from_pyobject(pyvalue, &value)
            cpp.godot_dictionary_set_item(ret, key, value)
    else:
        cpp.UtilityFunctions.push_error("a mapping is required, got %r" % type(p_obj))

    r_ret[0] = cpp.Variant(ret)


def as_array(data, dtype=None, itemshape=None, copy=None):
    """
    Interpret the input as Array
    """
    if dtype is None:
        dtype = np.object_

    nocopy_requested = copy is not None and not copy

    if not copy:
        if isinstance(data, numpy.ndarray):
            if data.dtype != dtype:
                if nocopy_requested:
                    cpp.UtilityFunctions.push_warning("Copy of %r will be made because data types do not match" % data)
                copy = True 
        else:
            if nocopy_requested:
                cpp.UtilityFunctions.push_warning("Copy of %r will be made because data types do not match" % data)
            copy = True

    return Array(data, dtype=dtype, itemshape=itemshape, copy=copy, can_cast=True)


class Array(numpy.ndarray):
    def __new__(subtype, data, **kwargs):
        cdef numpy.ndarray base

        dtype = kwargs.pop('dtype', np.object_)
        
        copy = kwargs.pop('copy', True)
        can_cast = kwargs.pop('can_cast', False)
        itemshape = kwargs.pop('itemshape', ())

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

        cdef object ret = PyArraySubType_NewFromBase(subtype, base)

        ret._itemshape = itemshape

        return ret


    def __array_finalize__(self, obj):
        ndim = self.ndim
        itemshape = self._itemshape
        itemsize = sum(itemshape)

        if ndim != ((len(itemshape)) + 1):
            if len(itemshape) > 0:
                self.shape = (self.size / itemsize,) + itemshape
            else:
                self.shape = (self.size,)


cdef tuple _vartype_to_dtype_itemshape = (
    None,
    (np.bool_,   ()),
    (np.int64,   ()),
    (np.float64, ()),
    (np.dtypes.StringDType, ()),
    (np.float64, (2,)), # vector2
    (np.int64,   (2,)),
    (np.float64, (4,)), # rect2
    (np.int64,   (4,)),
    (np.float64, (3,)), # vector3
    (np.int64,   (3,)),
    (np.float64, (3, 2)), # transform2d
    (np.float64, (4,)), # vector4
    (np.int64,   (4,)),
    (np.float64, (4,)), # plane
    (np.float64, (4,)), # quaternion
    (np.float64, (2, 3)), # aabb
    (np.float64, (3, 3)), # basis
    (np.float64, (4, 3)), # transform3d
    (np.float64, (4, 4)), # projection
    (np.float64, (4,)), # color
    (np.object_, ()), # stringname, must hold C++ string names
    (np.dtypes.StringDType, ()), # nodepath
    # All others are (np.object_, ()),
)


# NOTE: By default untyped Godot's Array conversions are to the ordinary
#       Python list. Typed Array conversions are always to `Array`.

# TODO: Convert typed numpy arrays to typed Godot arrays

cdef public object array_to_pyobject(const cpp.Array &p_arr):
    cdef int64_t size = p_arr.size(), i = 0

    cdef ret = PyList_New(size)
    cdef cpp.Variant item
    cdef object pyitem
    cdef int vartype

    for i in range(size):
        item = cpp.godot_array_get_item(p_arr, i)
        pyitem = variant_to_pyobject(item)

        ref.Py_INCREF(pyitem)
        PyList_SET_ITEM(ret, i, pyitem)

    dtype = np.object_
    itemshape = ()

    if p_arr.is_typed():
        vartype = p_arr.get_typed_builtin()
        if vartype > 0 and vartype < len(_vartype_to_dtype_itemshape):
            dtype, itemshape = _vartype_to_dtype_itemshape[vartype]
            cpp.UtilityFunctions.print("Detected %s typedarray %r" % (variant_type_to_str(<cpp.VariantType>vartype), dtype))

        # Array data has to be copied because it contains Variants that must be converted for Python
        return as_array(ret, dtype=dtype, itemshape=itemshape, copy=True)

    return ret


cdef public object variant_array_to_pyobject(const cpp.Variant &v):
    cdef cpp.Array ret = v.to_type[cpp.Array]()

    return array_to_pyobject(ret)


cdef public void array_from_pyobject(object p_obj, cpp.Array *r_ret) noexcept:
    cdef int64_t size, i
    cdef cpp.Array ret
    cdef cpp.Variant item
    cdef object pyitem

    # TODO: Typed arrays to typed arrays

    if PySequence_Check(p_obj):
        size = PySequence_Size(p_obj)
        ret.resize(size)
        for i, pyitem in enumerate(p_obj):
            variant_from_pyobject(pyitem, &item)
            cpp.godot_array_set_item(ret, i, item)
    else:
        cpp.UtilityFunctions.push_error("'list' or other sequence is required, got %r" % type(p_obj))

    r_ret[0] = ret


cdef void variant_array_from_pyobject(object p_obj, cpp.Variant *r_ret) noexcept:
    cdef int64_t size, i
    cdef cpp.Array ret
    cdef cpp.Variant item
    cdef object pyitem
    cdef cpp.String s

    # TODO: Typed arrays to typed arrays

    if PySequence_Check(p_obj):
        size = PySequence_Size(p_obj)
        ret.resize(size)
        for i, pyitem in enumerate(p_obj):
            variant_from_pyobject(pyitem, &item)
            cpp.godot_array_set_item(ret, i, item)

    else:
        cpp.UtilityFunctions.push_error("'list' or other sequence is required, got %r" % type(p_obj))

    r_ret[0] = cpp.Variant(ret)
