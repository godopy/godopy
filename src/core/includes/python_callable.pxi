cdef class PythonCallable(Callable):
    """
    Implements "Callable" GDExtension API (CallableCustomInfo, callable_custom_create2).
    """
    def __cinit__(self, *args, **kwargs):
        self.type_info = None
        self._type_info_opt = [0]*16
        self.__func__ = None
        self.__name__ = ''
        self.initialized = False


    def __init__(self, instance: Optional[object], func: typing.Callable | ExtensionMethod, type_info: Optional[Sequence] = None):
        """
        On the side of the Godot Engine creates a custom Callable object with all required callbacks.

        PythonCallable objects are used in Python ptrcalls and varcalls for all calls from the Engine to Python.
        """
        cdef GDExtensionCallableCustomInfo2 info

        cdef void *self_ptr = <PyObject *>self
        cdef void *func_ptr = <PyObject *>func
        cdef ExtensionMethod method

        self.__name__ = func.__name__
        self.__self__ = instance

        if isinstance(func, ExtensionMethod):
            method = func
            assert method.is_registered, "attempt to bind unregistered extension method"
            self.__func__ = method.__func__
            self.type_info = method.type_info

        elif callable(func):
            self.__func__ = func
            self.type_info = type_info

        else:
            raise ValueError("Python callable is required")

        if self.type_info is not None:
            make_optimized_type_info(self.type_info, self._type_info_opt)

        self.error_count = 0

        info.callable_userdata = self_ptr
        info.token = gdextension_token

        if instance is not None:
            info.object_id = instance.get_instance_id()

        info.call_func = &PythonCallable.call_callback
        info.is_valid_func = NULL
        info.free_func = &PythonCallable.free_callback
        info.hash_func = &PythonCallable.hash_callback
        info.equal_func = &PythonCallable.equal_callback
        info.less_than_func = &PythonCallable.less_than_callback
        info.to_string_func = &PythonCallable.to_string_callback
        info.get_argument_count_func = &PythonCallable.get_argument_count_callback

        ref.Py_INCREF(self)
        gdextension_interface_callable_custom_create2(self._godot_callable._native_ptr(), &info)
        self.initialized = True

    def __dealloc__(self):
        self.free()


    def __call__(self, *args, **kwargs):
        if self.__self__ is not None:
            return self.__func__(self.__self__, *args, **kwargs)
        else:
            return self.__func__(*args, **kwargs)


    def __str__(self):
        return self.__name__


    def __repr__(self):
        # cdef str args = "(%s)" % ', '.join(ai.name for ai in self.get_argument_info_list()[1:])
        cdef self_addr = <uint64_t><PyObject *>self
        cdef godot_addr = <uint64_t>self._godot_callable._native_ptr()

        if self.__self__ is not None:
            return "<Bound %s %s.%s of %r at %X[%X]>" % (self.__class__.__name__, self.__self__.__class__.__name__,
                                                         self.__name__, self.__self__, self_addr, godot_addr)
        else:
            return "<Unbound %s %s at %X[%X]>" % (self.__class__.__name__, self.__name__, self_addr, godot_addr)


    @staticmethod
    cdef void call_callback(void *callable_userdata, const (const void *) *p_args, int64_t p_count, void *r_return,
                            GDExtensionCallError *r_error) noexcept nogil:
        with gil:
            self = <object>callable_userdata
            _make_python_varcall(
                <PythonCallable>self,
                <const Variant **>p_args,
                p_count,
                <Variant *>r_return,
                r_error
            )


    @staticmethod
    cdef uint32_t hash_callback(void *callable_userdata) noexcept nogil:
        cdef uint32_t _hash
        with gil:
            return (<PythonCallable>callable_userdata).hash()


    cdef uint32_t hash(self) except -1:
        _hash = hash_murmur3_one_64(<uint64_t><PyObject *>self.__func__)
        if self.__self__ is not None:
            _hash =  hash_murmur3_one_64(<uint64_t><PyObject *>self.__self__, _hash)

        return _hash


    @staticmethod
    cdef uint8_t equal_callback(void *callable_userdata_a, void *callable_userdata_b) noexcept nogil:
        cdef uint64_t af, bf, ai, di
        with gil:
            af = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__func__
            bf = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__func__
            ai = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__self__
            bi = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__self__

            if af == bf:
                return ai == bi

            return False


    @staticmethod
    cdef uint8_t less_than_callback(void *callable_userdata_a, void *callable_userdata_b) noexcept nogil:
        cdef uint64_t af, bf, ai, di
        with gil:
            af = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__func__
            bf = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__func__
            ai = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__self__
            bi = <uint64_t><PyObject *>(<PythonCallable>callable_userdata_a).__self__

            if af == bf:
                return ai < bi

            return af < bf


    @staticmethod
    cdef void to_string_callback(void *callable_userdata, uint8_t *r_is_valid, void *r_out) noexcept nogil:
        with gil:
            self = <object>callable_userdata
            try:
                type_funcs.string_from_pyobject(repr(self), <String *>r_out)
                r_is_valid[0] = True
            except Exception as exc:
                print_error_with_traceback(exc)
                r_is_valid[0] = False


    @staticmethod
    cdef int64_t get_argument_count_callback(void *callable_userdata, uint8_t *r_is_valid) noexcept nogil:
        cdef int64_t result
        with gil:
            self = <object>callable_userdata
            try:
                result = (<PythonCallable>self).get_argument_count()
                r_is_valid[0] = True
                return result
            except Exception as exc:
                print_error_with_traceback(exc)
                r_is_valid[0] = False


    cdef int64_t get_argument_count(self) except -1:
        if self.__func__ is None:
            raise ValueError('%r function is not initialized' % self.__class__)

        # TODO: Support different Python callables
        if self.__self__ is not None:
            return self.__func__.__code__.co_argcount - 1
        else:
            return self.__func__.__code__.co_argcount


    @staticmethod
    cdef void free_callback(void *callable_userdata) noexcept nogil:
        with gil:
            self = <object>callable_userdata
            try:
                (<PythonCallable>self).free()
            except Exception as exc:
                print_error_with_traceback(exc)

    cdef int free(self) except -1:
        if self.initialized:
            ref.Py_DECREF(self)
            self.initialized = False
