cdef class VariantMethod:
    def __init__(self, object instance, method_name: Str) -> None:
        type_funcs.variant_from_pyobject(instance, &self._self_variant)

        self.__name__ = method_name
        self.__self__ = instance
        self._method = StringName(<const PyObject *>method_name)


    def __call__(self, *args):
        try:
            return _make_engine_varcall[VariantMethod](self, self._varcall, args)
        except Exception as exc:
            print_error_with_traceback(exc)


    def __repr__(self):
        class_name = '%s[%s.%s]' % (self.__class__.__name__, self.__self__.__class__.__name__, self.__name__)

        return "<%s.%s at 0x%016X>" % (self.__class__.__module__, class_name, <uint64_t><PyObject *>self)


    cdef void _varcall(self, const Variant **p_args, size_t p_count, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil:
        with nogil:
            gdextension_interface_variant_call(
                self._self_variant._native_ptr(),
                self._method._native_ptr(),
                <GDExtensionConstVariantPtr *>p_args,
                p_count,
                r_ret,
                r_error
            )
