cdef class MethodBind:
    def __cinit__(self, Object wrapper, str method_name, _methods_data=None):
        self._owner = wrapper._owner
        # print('GET MB %x %s %s' % (<uint64_t>self._owner, wrapper.__godot_class__.__name__, method_name))
        if _methods_data is not None:
            info = _methods_data.get(method_name)
        else:
            info = wrapper.__godot_class__._methods.get(method_name)
        # print("MB %s: %r" % (method_name, info))
        if info is None:
            raise AttributeError('Method %r not found in class %r' % (method_name, wrapper.__godot_class__.__name__))
        
        self.return_type = info['return_type']
        self._gde_mb = _gde_classdb_get_method_bind(
            StringName(wrapper.__godot_class__.__name__)._native_ptr(), StringName(method_name)._native_ptr(), info['hash'])


    cdef int _call_internal_nil_int_bool(self, int32_t p_arg1, bint p_arg2) except -1:
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *> \
            _gde_mem_alloc(2 * cython.sizeof(GDExtensionConstTypePtr))

        p_args[0] = &p_arg1
        p_args[1] = &p_arg2
        _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, NULL)
        _gde_mem_free(p_args)

        return 0


    cpdef object _call_internal(self, tuple args):
        cdef Variant gd_ret
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *> \
            _gde_mem_alloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = <Variant>args[i]
            p_args[i] = &arg

        cdef bint unknown_type_error = False
        with nogil:
            if self.return_type == 'Nil':
                _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, NULL)
            elif self.return_type == 'String':
                gd_ret = self._ptrcall_string(p_args)
            elif self.return_type == 'Variant':
                _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, &gd_ret)
            else:
                unknown_type_error = True
            _gde_mem_free(p_args)

        if unknown_type_error:
            _printerr("Don't know how to return %r types" % self.return_type)
            raise NotImplementedError("Don't know how to return %r types" % self.return_type)

        if self.return_type != 'Nil':
            return

        return gd_ret.pythonize()


    cdef Variant _ptrcall_string(self, GDExtensionConstTypePtr *p_args) noexcept nogil:
        cdef String gd_ret
        with nogil:
            _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, &gd_ret)
            return <Variant>gd_ret

    def __call__(self, *args):
        return self._call_internal(args)


cdef dict _utlity_functions = pickle.loads(_utility_function_data)

cdef class UtilityFunction:
    def __cinit__(self, str function_name):
        info = _utlity_functions.get(function_name, None)
        if info is None:
            raise NameError('Utility function %s not found' % function_name)
        # _print(info)
        self.return_type = info['return_type']
        self._gde_uf = \
            _gde_variant_get_ptr_utility_function(StringName(function_name)._native_ptr(), info['hash'])

    cpdef object _call_internal(self, tuple args):
        cdef Variant ret
        cdef Variant arg
        cdef size_t i
        cdef size_t size = len(args)
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *> \
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        for i in range(size):
            arg = <Variant>args[i]
            p_args[i] = &arg

        cdef bint unknown_type_error = False
        with nogil:
            if self.return_type == 'Nil':
                self._gde_uf(&ret, p_args, size)
            elif self.return_type == 'Variant':
                self._gde_uf(&ret, p_args, size)
            else:
                unknown_type_error = True
            _gde_mem_free(p_args)

        if unknown_type_error:
            _printerr("Don't know how to return %r types" % self.return_type)
            raise NotImplementedError("Don't know how to return %r types" % self.return_type)

        if self.return_type == 'Nil':
            return

        return ret.pythonize()

    def __call__(self, *args):
        return self._call_internal(args)
