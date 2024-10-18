cdef class _CallableBase:
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
        cdef Vector2i vector2i_arg
        cdef Rect2 rect2_arg
        cdef Rect2i rect2i_arg

        cdef StringName stringname_arg
        cdef NodePath nodepath_arg

        cdef PackedStringArray packed_string_array_arg

        cdef Object object_arg
        cdef Extension ext_arg
        cdef void *void_ptr_arg
        
        cdef double x, y, z, w
        cdef int32_t xi, yi, zi, wi

        cdef object pyarg

        # Optimized get_node for Python nodes
        if self.__name__ == 'get_node' and size == 1 and args[0] in _NODEDB:
            pyarg = _NODEDB[args[0]]
            # print("'get_node' shortcut for %r" % pyarg)
            return pyarg

        # TODO: Optimize
        for i in range(size):
            arg_type = self.type_info[i + 1]
            pyarg = args[i]
            
            if arg_type == 'bool':
                bool_arg = arg.booleanize()
                p_args[i] = &bool_arg
            elif arg_type == 'int' or arg_type == 'RID' or arg_type.startswith('enum:'):
                int_arg = pyarg
                p_args[i] = &int_arg
            elif arg_type == 'float':
                float_arg = pyarg
                p_args[i] = &float_arg
            elif arg_type == 'String':
                string_arg = <String>pyarg
                p_args[i] = &string_arg
            elif arg_type == 'Vector2':
                x, y = pyarg
                vector2_arg = Vector2(x, y)
                p_args[i] = &vector2_arg
            elif arg_type == 'Vector2i':
                xi, yi = pyarg
                vector2i_arg = Vector2i(xi, yi)
                p_args[i] = &vector2i_arg
            elif arg_type == 'Rect2':
                position, size = pyarg
                x, y = position
                z, w = size
                rect2_arg = Rect2(x, y, z, w)
                p_args[i] = &rect2_arg
            elif arg_type == 'Rect2i':
                position, size = pyarg
                xi, yi = position
                zi, wi = size
                rect2i_arg = Rect2i(xi, yi, zi, wi)
                p_args[i] = &rect2_arg
            elif arg_type == 'StringName':
                stringname_arg = <StringName>pyarg
                p_args[i] = &stringname_arg
            elif arg_type == 'NodePath':
                nodepath_arg = <NodePath>pyarg
                p_args[i] = &nodepath_arg
            elif arg_type == 'Variant':
                arg = Variant(<const PyObject *>pyarg)
                p_args[i] = &arg
            elif arg_type in _global_inheritance_info and isinstance(pyarg, Object):
                object_arg = <Object>pyarg
                p_args[i] = &object_arg._owner

            else:
                unknown_argtype_error = True
                break

        if unknown_argtype_error:
            gdextension_interface_mem_free(p_args)
            UtilityFunctions.printerr(
                "Don't know how to convert %r types, passed arg was: %r" % (arg_type, pyarg)
            )
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
        elif return_type == 'int' or return_type == 'RID' or return_type[6:] in _global_enum_info:
            self._ptr_call(&int_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>int_arg
        elif return_type == 'bool':
            self._ptr_call(&bool_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>bool_arg
        elif return_type == 'Vector2':
            self._ptr_call(&vector2_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>vector2_arg
        elif return_type == 'PackedStringArray':
            self._ptr_call(&packed_string_array_arg, <GDExtensionConstTypePtr *>p_args, size)
            arg = <Variant>packed_string_array_arg
        elif return_type in _global_inheritance_info:
            # print("Calling from %r with %r, receiving %s" % (self, args, return_type))
            self._ptr_call(&void_ptr_arg, <GDExtensionConstTypePtr *>p_args, size)
            object_arg = _OBJECTDB.get(<uint64_t>void_ptr_arg, None)
            # print("Process %s return value %r" % (return_type, object_arg))
            if object_arg is None and void_ptr_arg != NULL:
                object_arg = Object(return_type, from_ptr=<uint64_t>void_ptr_arg)
                # print("Created %s return value from pointer %X: %r" % (return_type, <uint64_t>void_ptr_arg, object_arg))
            gdextension_interface_mem_free(p_args)
            return object_arg
        else:
            unknown_type_error = True

        gdextension_interface_mem_free(p_args)

        if unknown_type_error:
            UtilityFunctions.printerr("Don't know how to return %r types. Returning None." % return_type)
            # raise NotImplementedError("Don't know how to return %r types" % return_type)
            return

        if return_type == 'Nil':
            return None

        return arg.pythonize()


    cdef void _ptr_call(self, GDExtensionTypePtr r_ret, GDExtensionConstTypePtr *p_args, size_t p_numargs) noexcept nogil:
        with gil:
            raise NotImplementedError("Base Callable Type: don't know how to ptrcall...")
