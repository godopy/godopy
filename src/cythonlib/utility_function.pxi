cdef dict _utlity_functions = pickle.loads(_utility_function_data)


cdef class UtilityFunction(Callable):
    def __init__(self, str function_name):
        info = _utlity_functions.get(function_name, None)
        if info is None:
            raise NameError('Utility function %s not found' % function_name)
        # builtins.print(function_name, info)
        self.type_info = info['type_info']
        cdef StringName name = StringName(function_name)
        cdef uint64_t _hash = info['hash']

        with nogil:
            self._godot_utility_function = \
                _gde_variant_get_ptr_utility_function(name._native_ptr(), _hash)


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        self._godot_utility_function(r_ret, p_args, p_numargs)
