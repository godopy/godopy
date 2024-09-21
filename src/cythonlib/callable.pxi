cdef class Callable:
    def __init__(self):
        raise NotImplementedError("Base class, cannot instantiate")


    def __call__(self, *args):
        return self._call_internal(args)


    cpdef object _call_internal(self, tuple args):
        cdef Variant arg
        cdef size_t i = 0
        cdef size_t size = len(args)

        cdef GDExtensionUninitializedTypePtr *p_args = <GDExtensionUninitializedTypePtr *> \
            _gde_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        cdef str return_type = 'Nil'
        cdef str arg_type = 'Nil'
        cdef bint unknown_argtype_error = False
        cdef bint unknown_type_error = False

        cdef int arg_typecode = 0

        cdef GDExtensionBool bool_arg
        cdef int64_t int_arg
        cdef real_t float_arg
        cdef String string_arg

        # TODO: Unpythonize and release the GIL, optimize
        for i in range(size):
            arg_type = self.type_info[i + 1]
            arg = Variant(args[i])
            arg_typecode = TYPEMAP_REVERSED.get(arg_type, 0)
            # with nogil:
            if arg_type == 'Variant':
                p_args[i] = &arg
            elif arg_type == 'bool':
                bool_arg = arg.booleanize()
                p_args[i] = &bool_arg
            elif arg_type == 'int':
                int_arg = args[i]
                p_args[i] = &int_arg
            elif arg_type == 'float':
                float_arg = args[i]
                p_args[i] = &float_arg
            elif arg_type == 'String':
                string_arg = <String>args[i]
                p_args[i] = &string_arg
            # elif arg_typecode > 0:
            #     print('ARG %d: %s %d' % (i, arg_type, arg_typecode))
            #     UtilityFunctions.print(arg)
            #     to_type_constructor[arg_typecode](&p_args[i], &arg)
            else:
                unknown_argtype_error = True
                break

        if unknown_argtype_error:
            _gde_mem_free(p_args)
            UtilityFunctions.printerr("Don't know how to convert %r types yet" % arg_type)
            raise NotImplementedError("Don't know how to return %r types" % arg_type)

        return_type = self.type_info[0]
        cdef Variant return_value
        cdef GDExtensionTypePtr type_return_value

        cdef int return_typecode = TYPEMAP_REVERSED.get(return_type, 0)

        # with nogil:
        if return_type == 'Nil':
            self._ptr_call(NULL, <GDExtensionConstTypePtr *>p_args, size)
        elif return_type == 'Variant':
            self._ptr_call(&return_value, <GDExtensionConstTypePtr *>p_args, size)
        elif return_typecode > 0:
            self._ptr_call(&type_return_value, <GDExtensionConstTypePtr *>p_args, size)
            if return_type == 'String':
                string_arg = deref(<String *>type_return_value)
            elif return_type == 'float':
                float_arg = deref(<real_t *>type_return_value)
            elif return_type == 'int':
                int_arg = deref(<int64_t *>type_return_value)
            elif return_type == 'bool':
                bool_arg = deref(<GDExtensionBool *>type_return_value)
            else:
                unknown_type_error = True
            # from_type_constructor[return_typecode](&return_value, type_return_value)
        else:
            unknown_type_error = True
        _gde_mem_free(p_args)

        if unknown_type_error:
            UtilityFunctions.printerr("Don't know how to return %r types" % return_type)
            raise NotImplementedError("Don't know how to return %r types" % return_type)

        if return_type == 'Nil':
            return

        return return_value.pythonize()


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        with gil:
            raise NotImplementedError("Base Callable Type: don't know how to ptrcall...")
