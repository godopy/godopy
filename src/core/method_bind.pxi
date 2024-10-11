cdef class MethodBind(Callable):
    def __init__(self, Object wrapper, str method_name):
        self._owner = wrapper._owner
        info = wrapper.__godot_class__.get_method_info(method_name)
        # print("MB %s: %r" % (method_name, info))
        if info is None:
            raise AttributeError('Method %r not found in class %r' % (method_name, wrapper.__godot_class__.__name__))
        
        self.type_info = info['type_info']
        cdef uint64_t _hash = info['hash']
        cdef StringName class_name = StringName(wrapper.__godot_class__.__name__)
        cdef StringName _method_name = StringName(method_name)
        with nogil:
            self._godot_method_bind = gdextension_interface_classdb_get_method_bind(
                class_name._native_ptr(), _method_name._native_ptr(), _hash)

    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        gdextension_interface_object_method_bind_ptrcall(self._godot_method_bind, self._owner, p_args, r_ret)