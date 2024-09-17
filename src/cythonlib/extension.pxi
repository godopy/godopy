cdef class Extension(gd.Object):
    @staticmethod
    cdef void* _create_callback(void *p_token, void *p_instance) noexcept nogil:
        with gil:
            print("EXT CREATE CALLBACK %x" % <int64_t>p_instance)
        return NULL

    @staticmethod
    cdef void _free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil:
        if p_binding:
            with gil:
                Extension._free_callback_gil(p_binding)

    @staticmethod
    cdef void _free_callback_gil(void *p_binding):
        print("EXT FREE CALLBACK %x" % <int64_t>p_binding)
        cdef Extension self = <object>p_binding
        ref.Py_DECREF(self)

    @staticmethod
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance,
                                             GDExtensionBool p_reference) noexcept nogil:
        return True

    def __init__(self):
        raise RuntimeError('Not supported, please use Extension.create')

    @staticmethod
    def create(object godot_class):
        global registry

        # self._binding_callbacks.create_callback = &Extension._create_callback
        # self._binding_callbacks.free_callback = &Extension._free_callback
        # self._binding_callbacks.reference_callback = &Extension._reference_callback

        if isinstance(godot_class, str):
            try:
                godot_class = registry[godot_class]
            except KeyError:
                raise NameError('Extension class %s not found' % godot_class)

        if not isinstance(godot_class, ExtensionClass):
            raise TypeError("Argument must be an ExtensionClass instance")
        if not godot_class.is_registered:
            raise RuntimeError('Extension class must be registered')

        cdef str class_name = godot_class.__name__

        cdef void *_owner = _gde_classdb_construct_object(StringName(class_name)._native_ptr())
        print('CONSTRUCT EXT OBJ %x' % <uint64_t>_owner)

        cdef Extension self = <Extension>_owner
        self._owner = _owner
        self.__godot_class__ = godot_class

        _gde_object_set_instance(self._owner, StringName(class_name)._native_ptr(), <void *><PyObject *>self)
        ref.Py_INCREF(self)
        print('SET BINDING TO %x' % <uint64_t><PyObject *>self)
        _gde_object_set_instance_binding(self._owner,
                                        StringName(class_name)._native_ptr(),
                                        <void *><PyObject *>self, &self._binding_callbacks)
        ref.Py_INCREF(self)
