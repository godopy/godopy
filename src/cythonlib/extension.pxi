cdef class Extension(gd.Object):
    cdef StringName _godot_class_name
    cdef StringName _godot_base_class_name

    cdef readonly object _wrapped

    def __init__(self, gd.Class base_class, ExtensionClass ext_class, bint notify=True):
        global registry

        self._binding_callbacks.create_callback = &Extension._create_callback
        self._binding_callbacks.free_callback = &Extension._free_callback
        self._binding_callbacks.reference_callback = &Extension._reference_callback

        if not isinstance(base_class, gd.Class):
            raise TypeError("godot.Class instance is required for 'ext_class', got %r" % type(base_class))

        if not isinstance(ext_class, ExtensionClass):
            raise TypeError("ExtensionClass instance is required for 'ext_class', got %r" % type(ext_class))

        if not ext_class.is_registered:
            raise RuntimeError('Extension class must be registered')

        self.__godot_class__ = ext_class

        cdef str class_name = ext_class.__name__
        self._godot_class_name = StringName(class_name)

        cdef str base_class_name = base_class.__name__
        self._godot_base_class_name = StringName(base_class_name)

        self._owner = _gde_classdb_construct_object(self._godot_base_class_name._native_ptr())

        # cdef gd.MethodBind mb = gd.MethodBind(self, 'notification')

        # if notify:
        #     mb._call_internal_nil_int_bool(0, False) # NOTIFICATION_POSTINITIALIZE

        ref.Py_INCREF(self) # DECREF in ExtensionClass._free
        _gde_object_set_instance(self._owner, self._godot_class_name._native_ptr(), <void *><PyObject *>self)

        ref.Py_INCREF(self)  # DECREF in Extension._free_callback
        _gde_object_set_instance_binding(
            self._owner,
            self._godot_class_name._native_ptr(),
            <void *><PyObject *>self,
            &self._binding_callbacks
        )

        class InnerExtensionObject:
            pass

        self._wrapped = InnerExtensionObject()
        cdef object wrapped_init = self.__godot_class__.method_bindings.get('__init__')
        if wrapped_init and callable(wrapped_init):
            wrapped_init(self._wrapped)


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
        with gil:
            print('REFERENCE CALLBACK EXT')
        return True
