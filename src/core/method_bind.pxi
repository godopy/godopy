cdef class MethodBind(GodotCppCallableBase):
    def __init__(self, Object wrapper, str method_name):
        self._owner = wrapper._owner
        self.__name__ = method_name
        self.__owner__ = wrapper

        info = wrapper.__godot_class__.get_method_info(method_name)
        # print("MB %s: %r" % (method_name, info))
        if info is None:
            raise AttributeError('Method %r not found in class %r'
                                 % (method_name, wrapper.__godot_class__.__name__))
        
        self.type_info = info['type_info']
        cdef uint64_t _hash = info['hash']
        cdef StringName class_name = StringName(wrapper.__godot_class__.__name__)
        cdef StringName _method_name = StringName(method_name)
        with nogil:
            self._godot_method_bind = gdextension_interface_classdb_get_method_bind(
                class_name._native_ptr(), _method_name._native_ptr(), _hash)

        # UtilityFunctions.print("Init MB %r" % self)


    def __repr__(self):
        class_name = '%s[%s.%s]' % (self.__class__.__name__, self.__owner__.__class__.__name__, self.__name__)
        return "<%s.%s at 0x%016X[0x%016X]>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                                         <uint64_t><PyObject *>self._godot_method_bind)


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args,
                        size_t p_numargs) noexcept nogil:
        with nogil:
            gdextension_interface_object_method_bind_ptrcall(self._godot_method_bind, self._owner, p_args, r_ret)


    # TODO: Auto-detect vararg methods and call them with method_bind_call and not method_bind_ptrcall
    def call(self, *args):
        cdef Variant ret
        cdef GDExtensionCallError err

        err.error = GDEXTENSION_CALL_OK

        cdef size_t i = 0, size = len(args)

        cdef Variant *p_args = <Variant *> gdextension_interface_mem_alloc(size * cython.sizeof(Variant))

        for i in range(size):
            p_args[i] = <Variant>args[i]

        with nogil:
            gdextension_interface_object_method_bind_call(self._godot_method_bind, self._owner,
                                                          <GDExtensionConstVariantPtr *>&p_args, size, &ret, &err)

        gdextension_interface_mem_free(p_args)

        if err.error != GDEXTENSION_CALL_OK:
            raise RuntimeError(ret.pythonize())

        return ret.pythonize()
