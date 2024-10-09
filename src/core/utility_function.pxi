cdef dict _global_utlity_function_info = pickle.loads(_global_utility_function_info__pickle)


cdef class UtilityFunction(Callable):
    def __init__(self, str function_name):
        info = _global_utlity_function_info.get(function_name, None)
        if info is None:
            raise NameError('Utility function %s not found' % function_name)
        # builtins.print(function_name, info)
        self.type_info = info['type_info']
        cdef StringName name = StringName(function_name)
        cdef uint64_t _hash = info['hash']

        with nogil:
            self._godot_utility_function = \
                gdextension_interface_variant_get_ptr_utility_function(name._native_ptr(), _hash)


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        self._godot_utility_function(r_ret, p_args, p_numargs)
