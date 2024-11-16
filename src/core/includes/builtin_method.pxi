def get_builtin_method_info(type_name):
    return _global_builtin_method_info[type_name]


cdef class BuiltinMethod:
    @staticmethod
    cdef BuiltinMethod new_with_selfptr(object instance, object method_name, void *selfptr):
        cdef BuiltinMethod self = BuiltinMethod.__new__(BuiltinMethod)

        self.__name__ = method_name
        self.__self__ = instance
        self._self_owner = selfptr

        cdef str type_name = instance.__class__.__name__

        type_method_info = get_builtin_method_info(type_name)
        info = type_method_info.get(method_name, None)
        if info is None:
           raise NameError('Builtin method %r not found' % method_name)

        self.type_info = info['type_info']
        make_optimized_type_info(self.type_info, self._type_info_opt)
        cdef PyGDStringName name = PyGDStringName(method_name)
        cdef uint64_t _hash = info['hash']
        cdef int type_id = str_to_variant_type(type_name)

        self._godot_builtin_method = gdextension_interface_variant_get_ptr_builtin_method(
            <GDExtensionVariantType>type_id, name.ptr(), _hash
        )

        # UtilityFunctions.print("Init BM %r" % self)

        return self

    def __init__(self):
        raise RuntimeError("%r classes cannot be instantiated directly" % self.__class__)


    def __call__(self, *args):
        try:
            return _make_engine_ptrcall[BuiltinMethod](self, self._ptrcall, args)
        except Exception as exc:
            print_error_with_traceback(exc)


    def __repr__(self):
        class_name = '%s[%s.%s]' % (self.__class__.__name__, self.__self__.__class__.__name__, self.__name__)
        return "<%s.%s at 0x%016X[0x%016X]>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self,
                                                <uint64_t><PyObject *>self._godot_builtin_method)


    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil:
        self._godot_builtin_method(self._self_owner, <GDExtensionConstTypePtr *>p_args, r_ret, p_numargs)
