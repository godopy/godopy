cdef class MethodBind(EngineCallableBase):
    def __init__(self, Object instance, str method_name):
        self._base = instance._owner
        self.__name__ = method_name
        self.__self__ = instance

        info = instance.__godot_class__.get_method_info(method_name)
        if info is None:
            raise AttributeError('Method %r not found in class %r'
                                 % (method_name, instance.__godot_class__.__name__))
        
        self.type_info = info['type_info']
        self.is_vararg = info['is_vararg']
        cdef uint64_t _hash = info['hash']
        cdef StringName class_name = StringName(instance.__godot_class__.__name__)
        cdef StringName _method_name = StringName(method_name)
        with nogil:
            self._godot_method_bind = gdextension_interface_classdb_get_method_bind(
                class_name._native_ptr(), _method_name._native_ptr(), _hash)

        # UtilityFunctions.print("Init MB %r" % self)


    def __call__(self, *args):
        if self.is_vararg:
            return _make_engine_varcall[MethodBind](self, self._varcall, args)
        else:
            return _make_engine_ptrcall[MethodBind](self, self._ptrcall, args)


    def __repr__(self):
        class_name = '%s[%s.%s]' % (self.__class__.__name__, self.__self__.__class__.__name__, self.__name__)
        return "<%s.%s at 0x%016X[0x%016X]>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                                                <uint64_t><PyObject *>self._godot_method_bind)


    cdef void _ptrcall(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args,
                       size_t p_numargs) noexcept nogil:
        with nogil:
            gdextension_interface_object_method_bind_ptrcall(self._godot_method_bind, self._base, p_args, r_ret)


    cdef void _varcall(self, const GDExtensionConstVariantPtr *p_args, size_t size,
                       GDExtensionUninitializedVariantPtr r_ret, GDExtensionCallError *r_error) noexcept nogil:
        with nogil:
            gdextension_interface_object_method_bind_call(self._godot_method_bind, self._base, p_args, size, r_ret, r_error)