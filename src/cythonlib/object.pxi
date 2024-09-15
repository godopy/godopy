cdef class GodotObject:
    @staticmethod
    cdef GodotObject from_ptr(void *ptr):
        cdef GodotSingleton self = GodotSingleton.__new__(GodotSingleton)
        self._owner = ptr

        return self

    @staticmethod
    cdef PyObject* _create_callback_gil(void *p_token, void *p_instance):
        print("CREATE CALLBACK %x" % <int64_t>p_instance)
        cdef GodotSingleton wrapper = GodotObject.from_ptr(p_instance)
        ref.Py_INCREF(wrapper)

        print("CREATED BINDING %x" % <int64_t><PyObject *>wrapper)
        return <PyObject *>wrapper

    @staticmethod
    cdef void _free_callback_gil(void *p_binding):
        print("FREE CALLBACK %x" % <int64_t>p_binding)
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

    def __init__(self, object godot_class):
        self._binding_callbacks.create_callback = &GodotObject._create_callback
        self._binding_callbacks.free_callback = &GodotObject._free_callback
        self._binding_callbacks.reference_callback = &GodotObject._reference_callback

        if not isinstance(godot_class, (GodotClass, str)):
            raise TypeError("'godot_class' argument must be a GodotClass instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, GodotClass) \
                                           else GodotClass(godot_class)
        cdef str class_name = self.__godot_class__.name
        self._owner = _gde_classdb_construct_object(StringName(class_name)._native_ptr())
        print("CONSTRUCTED OWNER %x" % <int64_t>self._owner)
        _gde_object_set_instance_binding(self._owner,
                                         StringName(class_name)._native_ptr(),
                                         <void *><PyObject *>self, &self._binding_callbacks)


cdef class GodotSingleton(GodotObject):
    def __init__(self, object godot_class):
        if not isinstance(godot_class, (GodotClass, str)):
            raise TypeError("'godot_class' argument must be a GodotClass instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, GodotClass) \
                                           else GodotClass(godot_class)
        cdef str class_name = self.__godot_class__.__name__
        self._owner = _gde_global_get_singleton(StringName(class_name)._native_ptr())
        print("AQUIRED SINGLETON OWNER %x" % <int64_t>self._owner)
