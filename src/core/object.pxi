cdef dict _OBJECTDB = {}


cdef public object object_to_pyobject(void *godot_object):
    if godot_object == NULL:
        return None

    cdef Object obj = _OBJECTDB.get(<uint64_t>godot_object, None)

    if obj is None and godot_object != NULL:
        obj = Object('Object', from_ptr=<uint64_t>godot_object)
        get_type = MethodBind(obj, 'get_type')

        obj.cast_to(get_type())

    return obj


# cdef public object variant_object_to_pyobject(Variant &v):
#     cdef _Object _godot_object = v.to_type[_Object]()

#     return object_to_pyobject(_godot_object._owner)


cdef public void *object_from_pyobject(object obj):
    if isinstance(obj, Object):
        return (<Object>obj)._owner

    return object_from_other_pyobject(obj)


cdef public Variant variant_object_from_pyobject(object obj):
    return Variant(object_from_pyobject(obj))


cdef void *object_from_other_pyobject(object obj):
    cdef PythonObject *gd_obj = PythonRuntime.get_singleton().python_object_from_pyobject(obj)
    return gd_obj._owner


cdef class Object:
    def __cinit__(self):
        self.is_singleton = False

    def __init__(self, object godot_class, uint64_t from_ptr=0):
        if not isinstance(godot_class, (Class, str)):
            raise TypeError("'godot_class' argument must be a Class instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, Class) else Class.get_class(godot_class)
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
                if from_ptr:
                    self._owner = <void *>from_ptr
                else:
                    self._owner = gdextension_interface_classdb_construct_object(_class_name._native_ptr())

        _OBJECTDB[self.owner_id()] = self

    def cast_to(self, object godot_class):
        if not isinstance(godot_class, (Class, str)):
            raise TypeError("'godot_class' argument must be a Class instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, Class) else Class.get_class(godot_class)

        cdef str class_name = self.__godot_class__.__name__
        cdef void *prev_owner

        if not ClassDB.get_singleton().class_exists(class_name):
            raise NameError('Class %r does not exist' % class_name)

        cdef StringName _class_name = StringName(class_name)

        if Engine.get_singleton().has_singleton(class_name):
            prev_owner = self._owner
            with nogil:
                self._owner = gdextension_interface_global_get_singleton(_class_name._native_ptr())
            if not self.is_singleton and self._owner != prev_owner:
                UtilityFunctions.push_warning("Object %r was cast to singleton object, previous owner was lost")
                if <uint64_t>prev_owner in _OBJECTDB:
                    del _OBJECTDB[<uint64_t>prev_owner]
                    _OBJECTDB[self.owner_id()] = self
            self.is_singleton = True


    def __repr__(self):
        class_name = self.__class__.__name__
        if self.__class__ is Object or class_name == 'Extension':
            class_name = '%s[%s]' % (self.__class__.__name__, self.__godot_class__.__name__)

        if self._ref_owner != NULL:
            return "<%s.%s object at 0x%016X[0x%016X[0x%016X]]>" % (
                self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                <uint64_t>self._owner, <uint64_t>self._ref_owner)

        return "<%s.%s object at 0x%016X[0x%016X]>" % (
            self.__class__.__module__, class_name, <uint64_t><PyObject *>self, <uint64_t>self._owner)

    def owner_hash(self):
        return self.owner_id()

    def owner_id(self):
        return <uint64_t>self._owner

    def ref_owner_id(self):
        return <uint64_t>self._ref_owner

    def ref_get_object(self):
        self._ref_owner = gdextension_interface_ref_get_object(self._owner)

    def ref_set_object(self):
        if self._ref_owner != NULL:
            gdextension_interface_ref_set_object(self._owner, self._ref_owner)


class Callable(Object):
    pass


class Signal(Object):
    pass
