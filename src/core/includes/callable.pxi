cdef class Callable:
    def __init__(self, object arg=None, object func=None):
        cdef StringName method
        cdef Variant v

        if isinstance(arg, Callable):
            self._godot_callable = GodotCppCallable((<Callable>arg)._godot_callable)
        elif not isinstance(arg, Object) and not isinstance(func, str):
            raise ValueError("Invalid %s() signature" % self.__class__.__name__)
        elif isinstance(arg, Object) and isinstance(func, str):
            type_funcs.string_name_from_pyobject(func, &method)
            type_funcs.variant_from_pyobject(arg, &v)
            self._godot_callable = GodotCppCallable.create(v, method)
        else:
            self._godot_callable = GodotCppCallable()

    def callv(self, *args):
        cdef Array cargs
        type_funcs.array_from_pyobject(args, &cargs)

        return type_funcs.variant_to_pyobject(self._godot_callable.callv(cargs))

    def __call__(self, *args):
        return self.callv(*args)

    @staticmethod
    cdef Callable from_cpp(const GodotCppCallable &p_val):
        cdef Callable self = Callable.__new__(Callable)
        self._godot_callable = p_val

        return self

    def as_custom(self):
        cdef void *_godot_callable = self._godot_callable._native_ptr()
        cdef void *ptr = gdextension_interface_callable_custom_get_userdata(_godot_callable, gdextension_token)

        if ptr == NULL:
            raise TypeError("%r object is not a custom Callable" % self.__class__)

        cdef PythonCallable obj = <object>ptr

        return obj


cdef public object callable_to_pyobject(const GodotCppCallable &p_callable):
    cdef object pycallable = None

    cdef void *ptr = gdextension_interface_callable_custom_get_userdata(p_callable._native_ptr(), gdextension_token)

    if ptr != NULL:
        pycallable = <object>ptr
        return pycallable
    else:
        return Callable.from_cpp(p_callable)


cdef public object variant_callable_to_pyobject(const Variant &v):
    cdef GodotCppCallable c = v.to_type[GodotCppCallable]()

    return callable_to_pyobject(c)


cdef public void callable_from_pyobject(object p_obj, GodotCppCallable *r_ret) noexcept:
    if isinstance(p_obj, Callable):
        r_ret[0] = (<Callable>p_obj)._godot_callable
    else:
        print_error("Expected 'Callable', got %r" % type(p_obj))

        r_ret[0] = GodotCppCallable()


cdef public void variant_callable_from_pyobject(object p_obj, Variant *r_ret) noexcept:
    cdef GodotCppCallable ret

    if isinstance(p_obj, Callable):
        ret = (<Callable>p_obj)._godot_callable
    else:
        print_error("Expected 'Callable', got %r" % type(p_obj))

        ret = GodotCppCallable()

    r_ret[0] = Variant(ret)
