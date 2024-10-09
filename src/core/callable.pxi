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
            gdextension_interface_mem_alloc(size * cython.sizeof(GDExtensionConstTypePtr))

        cdef str return_type = 'Nil'
        cdef str arg_type = 'Nil'
        cdef bint unknown_argtype_error = False
        cdef bint unknown_type_error = False

        cdef int arg_typecode = 0

        cdef GDExtensionBool bool_arg
        cdef int64_t int_arg
        cdef double float_arg
        cdef String string_arg

        cdef Vector2 vector2_arg
        cdef double x, y

        # TODO: Optimize
        for i in range(size):
            arg_type = self.type_info[i + 1]            
            if arg_type == 'Variant':
                arg = Variant(args[i])
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
            elif arg_type == 'Vector2':
                x, y = args[i]
                vector2_arg = Vector2(x, y)
                p_args[i] = &vector2_arg
            else:
                unknown_argtype_error = True
                break

        if unknown_argtype_error:
            gdextension_interface_mem_free(p_args)
            UtilityFunctions.printerr("Don't know how to convert %r types yet" % arg_type)
            raise NotImplementedError("Don't know how to convert %r types" % arg_type)

        return_type = self.type_info[0]

        if return_type == 'Nil':
            self._ptr_call(NULL, <GDExtensionConstTypePtr *>p_args, size)
        elif return_type == 'Variant':
            self._ptr_call(&arg, <GDExtensionConstTypePtr *>p_args, size)
        elif return_type == 'String':
            self._ptr_call(&string_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>string_arg
        elif return_type == 'float':
            self._ptr_call(&float_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>float_arg
        elif return_type == 'int':
            self._ptr_call(&int_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>int_arg
        elif return_type == 'bool':
            self._ptr_call(&bool_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>bool_arg
        elif return_type == 'Vector2':
            self._ptr_call(&vector2_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>vector2_arg
        else:
            unknown_type_error = True

        gdextension_interface_mem_free(p_args)

        if unknown_type_error:
            UtilityFunctions.printerr("Don't know how to return %r types. Returning None." % return_type)
            # raise NotImplementedError("Don't know how to return %r types" % return_type)
            return

        if return_type == 'Nil':
            return

        return arg.pythonize()


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        with gil:
            raise NotImplementedError("Base Callable Type: don't know how to ptrcall...")
