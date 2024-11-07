cdef class VariantStaticMethod:
    def __init__(self, object variant_type, str method_name):
        self.__name__ = method_name

        cdef VariantType vartype

        if isinstance(variant_type, type):
            vartype = type_funcs.pytype_to_variant_type(variant_type)
        elif isinstance(variant_type, int):
            vartype = <VariantType><int>variant_type
        else:
            raise ValueError("Expected 'type', integer or integer enum, got %r" % variant_type)

        self.__self__ = vartype
        self._method = StringName(<const PyObject *>method_name)

    def __call__(self, *args):
        try:
            return _make_engine_varcall[VariantStaticMethod](self, self._varcall, args)
        except Exception as exc:
            print_error_with_traceback(exc)


    def __repr__(self):
        class_name = '%s[%s.%s]' % (self.__class__.__name__, variant_type_to_str(self.__self__), self.__name__)

        return "<%s.%s at 0x%016X>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self)


    cdef void _varcall(self, const Variant **p_args, size_t p_count, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil:
        with nogil:
            gdextension_interface_variant_call_static(
                <GDExtensionVariantType>self.__self__,
                self._method._native_ptr(),
                <GDExtensionConstVariantPtr *>p_args,
                p_count,
                r_ret,
                r_error
            )
