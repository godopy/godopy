cdef dict _OBJECTDB = {}


cdef public object object_to_pyobject(void *p_godot_object):
    if p_godot_object == NULL:
        return None

    cdef uint64_t obj_id = <uint64_t>p_godot_object
    cdef Object obj = _OBJECTDB.get(obj_id, None)

    if obj is None and p_godot_object != NULL:
        from entry_point import get_config
        config = get_config()
        constructor = config.get('object_constructor')

        obj = constructor(from_ptr=obj_id)
        obj.cast_to(obj.get_godot_class_name())

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
    """
    Defines all Godot Engine's objects.

    Combined with MethodBind/ScriptMethod classes, provides a full
    implementation of GDExtension's Object interface.

    Implements following GDExtension API calls:
        in `__init__(from_ptr=0)`:
            if object is a singleton:
                `global_get_singleton`
            else:
                `classdb_construct_object2`
                `object_set_instance`

        in `destroy()`: `object_destroy`
        in `get_godot_class_name()`: `object_get_class_name`
        in `cast_to()`: `object_cast_to`
        in `get_instance_from_id()`: `object_get_instance_from_id`
        in `get_instance_id()`: `object_get_instance_id`
        in `has_script_method()`: `object_has_script_method`

    See MethodBind.__call__, MethodBind._varcall, MethodBind._ptrcall for
    `object_method_bind_call`/`object_method_bind_ptrcall` API implementation.

    See ScriptMethod.__call__(), ScriptMethod._varcall for
    `object_call_script_method` API implementation.

    `object_*_instance_binding` API is not implemented, no use-cases for it
    were found.
    """
    def __cinit__(self) -> None:
        self.is_singleton = False
        self._instance_set = False
        self._needs_cleanup = False

    def __init__(self, godot_class: Class | str = 'Object', *, from_ptr: uint64_t = 0) -> None:
        if not isinstance(godot_class, (Class, str)):
            raise TypeError("'godot_class' argument must be a Class instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, Class) else Class.get_class(godot_class)
        cdef PyStringName class_name = PyStringName(self.__godot_class__.__name__)

        if not ClassDB.get_singleton().class_exists(class_name._base):
            raise NameError('Class %r does not exist' % class_name)

        cdef void *self_ptr = <void *><PyObject *>self

        if from_ptr:
            self._owner = <void *>from_ptr
        elif Engine.get_singleton().has_singleton(class_name._base):
            self._owner = gdextension_interface_global_get_singleton(class_name.ptr())
            self.is_singleton = True
        else:
            self._owner = gdextension_interface_classdb_construct_object2(class_name.ptr())
            gdextension_interface_object_set_instance(self._owner, class_name.ptr(), self_ptr)

            ref.Py_INCREF(self)
            self._instance_set = True
            self._needs_cleanup = True

            notification = MethodBind(self, 'notification')
            notification(0, False) # NOTIFICATION_POSTINITIALIZE

        _OBJECTDB[<uint64_t>self._owner] = self
        gdtypes.add_object_type(self.__class__)


    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if self.__class__ is Object or class_name == 'Extension':
            class_name = '%s[%s]' % (self.__class__.__name__, self.__godot_class__.__name__)

        if self._ref_owner != NULL:
            return "<%s.%s object at 0x%016X[0x%016X[0x%016X]]>" % (
                self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                <uint64_t>self._owner, <uint64_t>self._ref_owner)

        return "<%s.%s object at 0x%016X[0x%016X]>" % (
            self.__class__.__module__, class_name, <uint64_t><PyObject *>self, <uint64_t>self._owner)


    def __dealloc__(self) -> None:
        self.destroy(_internal_call=True)


    def destroy(self, *, _internal_call=False) -> None:
        """
        Destroys an Object.
        """
        if self._owner != NULL and self._needs_cleanup:
            # Will call ExtensionClass._free_instance for Extension objects
            gdextension_interface_object_destroy(self._owner)

            if <uint64_t>self._owner in _OBJECTDB:
                del _OBJECTDB[<uint64_t>self._owner]

            self._owner = NULL

            if self._instance_set:
                ref.Py_DECREF(self)
                self._instance_set = False

        elif not _internal_call:
            raise TypeError("%r can not be destroyed")


    def get_godot_class_name(self) -> PyStringName:
        """
        Gets the class name of an Object from the Engine.
        """
        cdef StringName class_name
        gdextension_interface_object_get_class_name(self._owner, gdextension_library, &class_name)

        return type_funcs.string_name_to_pyobject(class_name)


    def _switch_class(self, cls: object) -> None:
        """
        Changes a class of an Object, a helper for Object.cast_to implementations with concrete Python classes
        """
        if isinstance(cls, Class) or hasattr(cls, '__godot_class__') and isinstance(cls.__godot_class__, Class):
            self.__class__ = cls
        else:
            raise TypeError("Expected a Godot class, got %r" % cls)


    def cast_to(self, object godot_class: Class | str) -> None:
        """
        Casts an Object to a different type.
        """
        if not isinstance(godot_class, (Class, str)):
            raise ValueError("'godot_class' argument must be a Class instance or a string")

        self.__godot_class__ = godot_class if isinstance(godot_class, Class) else Class.get_class(godot_class)
        godot_class_name = self.get_godot_class_name()

        if godot_class_name != self.__godot_class__.__name__:
            gdextension_interface_object_cast_to(self._owner, self.__godot_class__.get_tag())

        cdef PyStringName class_name = PyStringName(self.__godot_class__.__name__)
        cdef void *prev_owner

        if not ClassDB.get_singleton().class_exists(class_name._base):
            raise NameError('Class %r does not exist' % class_name)

        if Engine.get_singleton().has_singleton(class_name._base):
            prev_owner = self._owner
            self._owner = gdextension_interface_global_get_singleton(class_name.ptr())
            if not self.is_singleton and self._owner != prev_owner:
                UtilityFunctions.push_warning("Object %r was cast to singleton object, previous owner was lost")
                if <uint64_t>prev_owner in _OBJECTDB:
                    del _OBJECTDB[<uint64_t>prev_owner]
                    _OBJECTDB[self.owner_id()] = self
            self.is_singleton = True

    @staticmethod
    def get_instance_from_id(uint64_t id) -> Object:
        """
        Gets an Object by its instance ID.
        """
        cdef void *owner = gdextension_interface_object_get_instance_from_id(id)

        return object_to_pyobject(owner)

    def get_instance_id(self) -> gdtypes.ObjectID:
        """
        Gets the instance ID from an Object.
        """
        return type_funcs.ObjectID(gdextension_interface_object_get_instance_id(self._owner))

    def has_script_method(self, method_name: str) -> bool:
        """
        Checks if this object has a script with the given method.
        """
        cdef PyStringName method = PyStringName(method_name)

        return gdextension_interface_object_has_script_method(self._owner, method.ptr())

    def call_script_method(self, method_name: str, *args: Any) -> Any:
        """
        Call the given script method on this object.
        """
        cdef ScriptMethod method = ScriptMethod(self, method_name)
        return method(*args)

    def owner_hash(self):
        return self.owner_id()

    def owner_id(self):
        return <uint64_t>self._owner

    def ref_owner_id(self):
        return <uint64_t>self._ref_owner

    def ref_get_object(self):
        """
        Gets the Object from a reference. Sets the resulting pointer to `self._ref_owner`.
        """
        self._ref_owner = gdextension_interface_ref_get_object(self._owner)

    def ref_set_object(self):
        """
        Sets the Object referred to by a reference.
        """
        if self._ref_owner != NULL:
            gdextension_interface_ref_set_object(self._owner, self._ref_owner)
