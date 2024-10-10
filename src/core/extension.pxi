cdef class Extension(Object):
    cdef StringName _godot_class_name
    cdef StringName _godot_base_class_name

    cdef readonly object _wrapped

    def __init__(self, ExtensionClass ext_class, Class base_class, bint notify=False, bint from_callback=False):
        if not isinstance(base_class, Class):
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

        self._owner = gdextension_interface_classdb_construct_object(self._godot_base_class_name._native_ptr())

        if notify:
            notification = MethodBind(self, 'notification')
            notification(0, False) # NOTIFICATION_POSTINITIALIZE

        ref.Py_INCREF(self) # DECREF in ExtensionClass._free
        gdextension_interface_object_set_instance(self._owner, self._godot_class_name._native_ptr(), <void *><PyObject *>self)

        class InnerExtensionObject:
            pass

        self._wrapped = InnerExtensionObject()
        self._wrapped.__godot_object__ = self
        cdef object wrapped_init = self.__godot_class__.python_method_bindings.get('__init__')
        if wrapped_init and callable(wrapped_init):
            wrapped_init(self._wrapped)

        print("INITIALIZED EXT OBJ %r %s %x" % (self, self.__godot_class__.__name__, <uint64_t>self._owner))


    cpdef destroy(self):
        # Will call ExtensionClass._free
        gdextension_interface_object_destroy(self._owner)

        self._owner = NULL
