cdef class MethodBindRet:
    def __cinit__(self, Object wrapper, str method_name, GDExtensionInt method_hash, returning='Variant'):
        self._owner = wrapper._owner
        print('GET MB %x %s %s' % (<uint64_t>self._owner, wrapper.__godot_class__.__name__, method_name))
        self.returning_type = returning
        self._gde_mb = _gde_classdb_get_method_bind(
            StringName(wrapper.__godot_class__.__name__)._native_ptr(), StringName(method_name)._native_ptr(), method_hash)

    cpdef object _call_internal(self, tuple args):
        cdef Variant gd_ret
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            if self.returning_type == 'String':
                gd_ret = self._ptrcall_string(p_args)
            else:
                _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, &gd_ret)
            _gde_mem_free(p_args)

        return gd_ret.pythonize()

    cdef Variant _ptrcall_string(self, GDExtensionConstTypePtr *p_args) noexcept nogil:
        cdef String gd_ret
        with nogil:
            _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, &gd_ret)
            return <Variant>gd_ret

    def __call__(self, *args):
        return self._call_internal(args)


cdef class MethodBindNoRet(MethodBindRet):
    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(len(args) * cython.sizeof(GDExtensionConstTypePtr))
        cdef int i
        for i in range(len(args)):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            _gde_object_method_bind_ptrcall(self._gde_mb, self._owner, p_args, NULL)
            _gde_mem_free(p_args)

cdef class UtilityFunctionRet:
    def __cinit__(self, str function_name, GDExtensionInt function_hash):
        self._gde_uf = \
            _gde_variant_get_ptr_utility_function(StringName(function_name)._native_ptr(), function_hash)

    cpdef object _call_internal(self, tuple args):
        cdef Variant ret
        cdef Variant arg
        cdef int i
        cdef int size = len(args)
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        for i in range(size):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            self._gde_uf(&ret, p_args, size)
            _gde_mem_free(p_args)

        return pyobject_from_variant(ret)

    def __call__(self, *args):
        return self._call_internal(args)


cdef class UtilityFunctionNoRet(UtilityFunctionRet):
    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef int i
        cdef int size = len(args)
        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *>\
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        for i in range(size):
            arg = variant_from_pyobject(args[i])
            p_args[i] = &arg

        with nogil:
            self._gde_uf(NULL, p_args, size)
            _gde_mem_free(p_args)
