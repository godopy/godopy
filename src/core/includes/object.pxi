cdef dict _OBJECTDB = {}


cdef public object object_to_pyobject(void *p_godot_object):
    if p_godot_object == NULL:
        return None

    cdef uint64_t obj_id = <uint64_t>p_godot_object
    cdef Object obj = _OBJECTDB.get(obj_id, None)

    if obj is None and p_godot_object != NULL:
        from godot import classdb
        cls = classdb.Object

        obj = cls(from_ptr=obj_id)
        obj.cast_to(obj.get_class())

    return obj


cdef public object variant_object_to_pyobject(const Variant &v):
    cdef GodotCppObject *o = v.to_type[GodotCppObjectPtr]()

    return object_to_pyobject(o._owner)


cdef public void object_from_pyobject(object p_obj, void **r_ret) noexcept:
    if isinstance(p_obj, Object):
        r_ret[0] = (<Object>p_obj)._owner
    else:
        object_from_other_pyobject(p_obj, r_ret)


cdef public void cppobject_from_pyobject(object p_obj, GodotCppObject **r_ret) noexcept:
    cdef void *godot_object

    if isinstance(p_obj, Object):
        godot_object = (<Object>p_obj)._owner
    else:
        object_from_other_pyobject(p_obj, &godot_object)

    cdef GodotCppObject *o = get_object_instance_binding(godot_object)

    r_ret[0] = o


cdef public void variant_object_from_pyobject(object p_obj, Variant *r_ret) noexcept:
    cdef void *godot_object
    object_from_pyobject(p_obj, &godot_object)

    cdef GodotCppObject *o = get_object_instance_binding(godot_object)
    r_ret[0] = Variant(o)


cdef void object_from_other_pyobject(object obj, void **r_ret) noexcept nogil:
    cdef PythonObject *gd_obj = PythonRuntime.get_singleton().python_object_from_pyobject(obj)
    r_ret[0] = gd_obj._owner


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

        if hasattr(self.__class__, '__godot_class__'):
            # TODO: gdextension should never import godot, refactor
            from godot import classdb
            self.__class__ = getattr(classdb, self.__godot_class__.__name__)

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


# TODO: Implement 'special' *Engine* object that has customized
#       'register_singleton' and 'register_script_language' methods
#       that would set 'is_singleton' to true.
