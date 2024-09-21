cdef class Callable:
    def __init__(self):
        raise NotImplementedError("Base class, cannot instantiate")


    def __call__(self, *args):
        return self._call_internal(args)


    cpdef object _call_internal(self, tuple args):
        return
        cdef Variant arg
        cdef size_t i = 0
        cdef size_t size = len(args)

        cdef GDExtensionConstTypePtr *p_args = <GDExtensionConstTypePtr *> \
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        cdef str return_type = 'Nil'
        cdef str arg_type = 'Nil'
        cdef bint unknown_argtype_error = False
        cdef bint unknown_type_error = False

        cdef int arg_typecode = 0

        # TODO: Unpythonize and release the GIL
        for i in range(size):
            arg_type = self.type_info[i + 1]
            arg = <Variant>args[i]
            arg_typecode = TYPEMAP_REVERSED.get(arg_type, 0)
            with nogil:
                if arg_type == 'Variant':
                    pass
                elif arg_typecode > 0:
                    to_type_constructor[arg_typecode](<GDExtensionUninitializedTypePtr>p_args[i], &arg)
                else:
                    unknown_argtype_error = True
                    break

                p_args[i] = &arg

        return_type = self.type_info[0]
        cdef Variant return_value
        cdef GDExtensionTypePtr type_return_value

        cdef int return_typecode = TYPEMAP_REVERSED.get(return_type, 0)

        with nogil:
            if return_type == 'Nil':
                self._ptr_call(NULL, p_args, size)
            elif return_type == 'Variant':
                self._ptr_call(&return_value, p_args, size)
            elif return_typecode > 0:
                self._ptr_call(type_return_value, p_args, size)
                from_type_constructor[return_typecode](&return_value, type_return_value)
            else:
                unknown_type_error = True
            _gde_mem_free(p_args)

        if unknown_argtype_error:
            _printerr("Don't know how to convert %r types yet" % arg_type)
            raise NotImplementedError("Don't know how to return %r types" % self.return_type)

        if unknown_type_error:
            _printerr("Don't know how to return %r types" % return_type)
            raise NotImplementedError("Don't know how to return %r types" % return_type)

        if return_type == 'Nil':
            return

        return return_value.pythonize()


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        with gil:
            raise NotImplementedError("Base Callable Type: don't know how to ptrcall...")
