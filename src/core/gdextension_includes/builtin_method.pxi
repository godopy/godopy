cdef dict _BUILTIN_METHODDB = {}


def get_builtin_method_info(type_name):
    cdef bytes mi_pickled

    if type_name not in _BUILTIN_METHODDB:
        mi_pickled = _global_builtin_method_info__pickles.get(type_name, None)
        _BUILTIN_METHODDB[type_name] = pickle.loads(mi_pickled) if mi_pickled is not None else {}

    return _BUILTIN_METHODDB[type_name]


cdef class BuiltinMethod(CallableBase):
    @staticmethod
    cdef BuiltinMethod new_with_baseptr(object owner, object method_name, void *_base):
        cdef BuiltinMethod self = BuiltinMethod.__new__(BuiltinMethod)

        self.__name__ = method_name
        self._base = _base

        self.__owner__ = owner

        cdef str type_name = owner.__class__.__name__

        type_method_info = get_builtin_method_info(type_name)
        info = type_method_info.get(method_name, None)
        if info is None:
           raise NameError('Builtin method %r not found' % method_name)

        self.type_info = info['type_info']
        self.is_vararg = False
        cdef StringName name = StringName(<const PyObject *>method_name)
        cdef uint64_t _hash = info['hash']
        cdef int type_id = str_to_variant_type(type_name)

        with nogil:
            self._godot_builtin_method = gdextension_interface_variant_get_ptr_builtin_method(
                <GDExtensionVariantType>type_id, name._native_ptr(), _hash
            )

        # UtilityFunctions.print("Init BM %r" % self)

        return self

    def __init__(self):
        raise RuntimeError("%r classes cannot be instantiated directly" % self.__class__)


    def __repr__(self):
        class_name = '%s[%s.%s]' % (self.__class__.__name__, self.__owner__.__class__.__name__, self.__name__)
        return "<%s.%s at 0x%016X[0x%016X]>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                                                <uint64_t><PyObject *>self._godot_builtin_method)


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        self._godot_builtin_method(self._base, p_args, r_ret, p_numargs)
