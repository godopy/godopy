cdef class GodotObject:
    # cdef void* _owner
    # cdef GDExtensionInstanceBindingCallbacks _binding_callbacks
    # cdef StringName _class_name
    # cdef readonly str __godot_class__

    @staticmethod
    cdef GodotObject from_ptr(void *ptr):
        cdef GodotSingleton self = GodotSingleton.__new__(GodotSingleton)
        self._owner = ptr

        return self

    @staticmethod
    cdef PyObject* _create_callback_gil(void *p_token, void *p_instance):
        print("CREATE CALLBACK", <int64_t>p_instance)
        cdef GodotSingleton wrapper = GodotObject.from_ptr(p_instance)
        ref.Py_INCREF(wrapper)

        print("CREATED BINDING", <int64_t><PyObject *>wrapper)
        return <PyObject *>wrapper

    @staticmethod
    cdef void _free_callback_gil(void *p_binding):
        print("FREE CALLBACK", <int64_t>p_binding)
        cdef GodotSingleton wrapper = <object>p_binding
        ref.Py_DECREF(wrapper)

    @staticmethod
    cdef void* _create_callback(void *p_token, void *p_instance) noexcept nogil:
        with gil:
            return <void *>GodotSingleton._create_callback_gil(p_token, p_instance)

    @staticmethod
    cdef void _free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil:
        if p_binding:
            with gil:
                GodotSingleton._free_callback_gil(p_binding)

    @staticmethod
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance,
                                             GDExtensionBool p_reference) noexcept nogil:
        return True

    def __init__(self, str class_name):
        self._binding_callbacks.create_callback = &GodotObject._create_callback
        self._binding_callbacks.free_callback = &GodotObject._free_callback
        self._binding_callbacks.reference_callback = &GodotObject._reference_callback

        self.__godot_class__ = class_name
        self._class_name = stringname_from_str(class_name)
        self._owner = _gde_classdb_construct_object(self._class_name._native_ptr())
        _gde_object_set_instance_binding(self._owner,
                                         self._class_name._native_ptr(),
                                         <void *><PyObject *>self, &self._binding_callbacks)


cdef class GodotSingleton(GodotObject):
    # cdef GDExtensionObjectPtr _gde_so
    # cdef void* singleton

    def __init__(self, str class_name):
        self._binding_callbacks.create_callback = &GodotObject._create_callback
        self._binding_callbacks.free_callback = &GodotObject._free_callback
        self._binding_callbacks.reference_callback = &GodotObject._reference_callback

        self.__godot_class__ = class_name
        self._class_name = stringname_from_str(class_name)
        self._gde_so = _gde_global_get_singleton(self._class_name._native_ptr())
        self.singleton = _gde_object_get_instance_binding(self._gde_so, gdextension_token, &self._binding_callbacks)
