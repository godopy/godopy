cdef public class Object [object GDPy_Object, type GDPy_ObjectType]:
    def __cinit__(self):
        self.is_singleton = False

    def __init__(self, object godot_class):
        if not isinstance(godot_class, (Class, str)):
            raise TypeError("'godot_class' argument must be a Class instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, Class) \
                                           else Class.get_class(godot_class)
        cdef str class_name = self.__godot_class__.__name__

        if not ClassDB.get_singleton().class_exists(class_name):
            raise NameError('Class %r does not exist' % class_name)

        cdef StringName _class_name = StringName(class_name)

        if Engine.get_singleton().has_singleton(class_name):
            with nogil:
                self._owner = gdextension_interface_global_get_singleton(_class_name._native_ptr())
            self.is_singleton = True
        else:
            if not ClassDB.get_singleton().can_instantiate(class_name):
                raise TypeError('Class %r can not be instantiated' % class_name)

            with nogil:
                self._binding_callbacks.create_callback = &Object._create_callback
                self._binding_callbacks.free_callback = &Object._free_callback
                self._binding_callbacks.reference_callback = &Object._reference_callback

                self._owner = gdextension_interface_classdb_construct_object(_class_name._native_ptr())
                gdextension_interface_object_set_instance_binding(
                    self._owner, _class_name._native_ptr(), <void *><PyObject *>self, &self._binding_callbacks)

    @staticmethod
    cdef Object from_ptr(void *ptr):
        cdef Object self = Object.__new__(Object)
        self._owner = ptr

        return self

    @staticmethod
    cdef PyObject* _create_callback_gil(void *p_token, void *p_instance):
        print("CREATE CALLBACK %x" % <int64_t>p_instance)
        cdef Object wrapper = Object.from_ptr(p_instance)
        ref.Py_INCREF(wrapper)

        return <PyObject *>wrapper

    @staticmethod
    cdef void _free_callback_gil(void *p_binding):
        print("FREE CALLBACK %x" % <int64_t>p_binding)
        cdef Object wrapper = <object>p_binding
        ref.Py_DECREF(wrapper)

    @staticmethod
    cdef void* _create_callback(void *p_token, void *p_instance) noexcept nogil:
        with gil:
            return <void *>Object._create_callback_gil(p_token, p_instance)

    @staticmethod
    cdef void _free_callback(void *p_token, void *p_instance, void *p_binding) noexcept nogil:
        if p_binding:
            with gil:
                Object._free_callback_gil(p_binding)

    @staticmethod
    cdef GDExtensionBool _reference_callback(void *p_token, void *p_instance,
                                             GDExtensionBool p_reference) noexcept nogil:
        return True