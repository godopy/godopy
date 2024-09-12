cdef class GodotExtension(godot.GodotObject):
    @staticmethod
    cdef void* _create_callback(void *p_token, void *p_instance) noexcept nogil:
        return NULL

    @staticmethod
    cdef void _free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil:
        if p_binding:
            with gil:
                GodotExtension._free_callback_gil(p_binding)

    @staticmethod
    cdef void _free_callback_gil(void *p_binding):
        print("EXT FREE CALLBACK", <int64_t>p_binding)
        cdef GodotExtension self = <object>p_binding
        ref.Py_DECREF(self)

    @staticmethod
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance,
                                             GDExtensionBool p_reference) noexcept nogil:
        return True

    def __init__(self, str class_name):
        self._binding_callbacks.create_callback = &GodotExtension._create_callback
        self._binding_callbacks.free_callback = &GodotExtension._free_callback
        self._binding_callbacks.reference_callback = &GodotExtension._reference_callback

        self.__godot_class__ = class_name
        self._owner = _gde_classdb_construct_object(StringName(class_name)._native_ptr())
        _gde_object_set_instance(self._owner, StringName(class_name)._native_ptr(), <void *><PyObject *>self)
        ref.Py_INCREF(self)
        _gde_object_set_instance_binding(self._owner,
                                         StringName(class_name)._native_ptr(),
                                         <void *><PyObject *>self, &self._binding_callbacks)
