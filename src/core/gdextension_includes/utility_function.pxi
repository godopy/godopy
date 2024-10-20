cdef class UtilityFunction(CallableBase):
    def __init__(self, str function_name):
        self.__name__ = function_name

        info = _global_utility_function_info.get(function_name, None)
        if info is None:
            raise NameError('Utility function %r not found' % function_name)

        self.type_info = info['type_info']
        self.is_vararg = False
        cdef StringName name = StringName(function_name)
        cdef uint64_t _hash = info['hash']

        with nogil:
            self._godot_utility_function = \
                gdextension_interface_variant_get_ptr_utility_function(name._native_ptr(), _hash)

        # UtilityFunctions.print("Init UF %r" % self)


    def __repr__(self):
        class_name = '%s[%s]' % (self.__class__.__name__, self.__name__)
        return "<%s.%s at 0x%016X[0x%016X]>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                                                <uint64_t><PyObject *>self._godot_utility_function)


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        self._godot_utility_function(r_ret, p_args, p_numargs)
