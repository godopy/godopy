cdef class Extension(gd.Object):
    cdef StringName _godot_class_name

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

    def __init__(self, object godot_class, bint from_callback=False):
        global registry

        ref.Py_INCREF(self)
        # self._owner = <void *><PyObject *>self
        print("INIT %x" % <uint64_t>self._owner)

        self._binding_callbacks.create_callback = &Extension._create_callback
        self._binding_callbacks.free_callback = &Extension._free_callback
        self._binding_callbacks.reference_callback = &Extension._reference_callback

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
        print('EXT CLASS NAME %s' % class_name)
        self._godot_class_name = StringName(class_name)
        if not from_callback:
            self._owner = _gde_classdb_construct_object(self._godot_class_name._native_ptr())

            print('GOT WRAPPER %x' % <uint64_t>self._owner)

        # cdef Extension self = <Extension>_owner
        # print('CONVERTED TO SELF %r' % self)
        # ref.Py_INCREF(self)
        # self._owner = _owner
        self.__godot_class__ = godot_class

        # ref.Py_INCREF(self)
        # _gde_object_set_instance(self._owner, _class_name._native_ptr(), <void *><PyObject *>self)

        # print('SET BINDING TO %x' % <uint64_t><PyObject *>self)
        # _gde_object_set_instance_binding(
        #     self._owner,
        #     self._godot_class_name._native_ptr(),
        #     self._owner,
        #     &self._binding_callbacks
        # )
        # ref.Py_INCREF(self)
