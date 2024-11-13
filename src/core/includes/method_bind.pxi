
def _engine_register_singleton(MethodBind self, class_name, Object singleton_obj) -> None:
    singleton_obj.is_singleton = True

    _make_engine_ptrcall[MethodBind](self, self._ptrcall, (class_name, singleton_obj))

_engine_register_singleton.makes_ptrcall = True


def _engine_register_script_language(MethodBind self, Object language_obj) -> int:
    language_obj.is_singleton = True

    return _make_engine_ptrcall[MethodBind](self, self._ptrcall, (language_obj,))

_engine_register_script_language.makes_ptrcall = True


def _xml_parser_open_buffer(MethodBind self, type_funcs.Buffer buffer) -> int:
    return gdextension_interface_xml_parser_open_buffer(self._self_owner, buffer.ptr, buffer.size)


def _file_access_store_buffer(MethodBind self, type_funcs.Buffer buffer) -> None:
    gdextension_interface_file_access_store_buffer(self._self_owner, buffer.ptr, buffer.size)


def _file_access_get_buffer(MethodBind self, type_funcs.Buffer buffer, size_t length) -> int:
    return gdextension_interface_file_access_get_buffer(self._self_owner, buffer.ptr, length)


def _image_ptrw(MethodBind self) -> type_funcs.Pointer:
    cdef uint8_t *ptr = gdextension_interface_image_ptrw(self._self_owner)

    return type_funcs.Pointer.create(ptr)


def _image_ptr(MethodBind self) -> type_funcs.Pointer:
    cdef const uint8_t *ptr = gdextension_interface_image_ptr(self._self_owner)

    return type_funcs.Pointer.create(ptr)


cdef void thread_pool_group_func(void *_data, uint32_t n) noexcept nogil:
    with gil:
        data_tuple = <object>_data
        func, data = data_tuple
        func(data, n)

        ref.Py_DECREF(data_tuple)

cdef void thread_pool_func(void *_data) noexcept nogil:
    with gil:
        data_tuple = <object>_data
        func, data = data_tuple
        func(data)

        ref.Py_DECREF(data_tuple)


def _worker_thread_pool_add_group_task(MethodBind self, func, userdata, int elements, int tasks,
                                       bint high_priority, description) -> int:
    cdef String descr = String(<const PyObject *>description)
    cdef tuple _data = (func, userdata)
    ref.Py_INCREF(_data)

    return gdextension_interface_worker_thread_pool_add_native_group_task(
        self._self_owner,
        &thread_pool_group_func,
        <void *><PyObject *>_data,
        elements,
        tasks,
        high_priority,
        descr._native_ptr()
    )


def _worker_thread_pool_add_task(MethodBind self, func, userdata, bint high_priority, description) -> int:
    cdef String descr = String(<const PyObject *>description)
    cdef tuple _data = (func, userdata)
    ref.Py_INCREF(_data)

    return gdextension_interface_worker_thread_pool_add_native_task(
        self._self_owner,
        &thread_pool_func,
        <void *><PyObject *>_data,
        high_priority,
        descr._native_ptr()
    )


cdef dict special_method_calls = {
    'Engine::register_singleton': _engine_register_singleton,
    'Engine::register_script_language': _engine_register_script_language,
    'XMLParser::open_buffer': _xml_parser_open_buffer,
    'FileAccess::store_buffer': _file_access_store_buffer,
    'FileAccess::get_buffer': _file_access_get_buffer,
    'Image::ptrw': _image_ptrw,
    'Image::ptr': _image_ptr,
    'WorkerThreadPool::add_group_task': _worker_thread_pool_add_group_task,
    'WorkerThreadPool::add_task': _worker_thread_pool_add_task
}


cdef class MethodBind:
    def __init__(self, Object instance, method_name: Str) -> None:
        self.__name__ = method_name
        self.__self__ = instance
        self._self_owner = instance._owner

        self.key = "%s::%s" % (self.__self__.__godot_class__.__name__, self.__name__)
        self.func = special_method_calls.get(self.key, None)

        cdef uint64_t _hash
        cdef PyGDStringName class_name, _method_name

        if self.func is None or getattr(self.func, 'makes_ptrcall', False):
            info = instance.__godot_class__.get_method_info(method_name)
            if info is None:
                msg = 'Method %r not found in class %r' % (method_name, instance.__godot_class__.__name__)
                raise AttributeError(msg)
            
            self.type_info = info['type_info']
            make_optimized_type_info(self.type_info, self._type_info_opt)
            self.is_vararg = info['is_vararg']
            _hash = info['hash']
            class_name = PyGDStringName(instance.__godot_class__.__name__)
            _method_name = PyGDStringName(method_name)

            self._godot_method_bind = gdextension_interface_classdb_get_method_bind(
                class_name.ptr(),
                _method_name.ptr(),
                _hash
            )
        else:
            self.type_info = ()
            self.is_vararg = False
            self._godot_method_bind = NULL

        # UtilityFunctions.print("Init MB %r" % self)

    def __str__(self):
        return self.key


    def __repr__(self):
        class_name = '%s[%s]' % (self.__class__.__name__, self.key)
        cdef uint64_t self_addr =  <uint64_t><PyObject *>self
        cdef uint64_t mb_addr = <uint64_t>self._godot_method_bind

        if self.func is not None:
            return "<%s.%s at 0x%016X>" % (self.__class__.__module__, class_name, self_addr)
        else:
            return "<%s.%s at 0x%016X[0x%016X]>" % (self.__class__.__module__, class_name, self_addr, mb_addr)


    def __call__(self, *args):
        """
        Calls a method on an Object.
        """
        try:
            if self.func is not None:
                return self.func(self, *args)
            elif self.is_vararg:
                return _make_engine_varcall[MethodBind](self, self._varcall, args)
            else:
                return _make_engine_ptrcall[MethodBind](self, self._ptrcall, args)
        except Exception as exc:
            print_error_with_traceback(exc)


    cdef void _ptrcall(self, void *r_ret, const void **p_args, size_t p_numargs) noexcept nogil:
        """
        Calls a method on an Object (using a "ptrcall").
        """
        with nogil:
            gdextension_interface_object_method_bind_ptrcall(self._godot_method_bind, self._self_owner, p_args, r_ret)


    cdef void _varcall(self, const Variant **p_args, size_t size, Variant *r_ret,
                       GDExtensionCallError *r_error) noexcept nogil:
        """
        Calls a method on an Object.
        """
        with nogil:
            gdextension_interface_object_method_bind_call(
                self._godot_method_bind,
                self._self_owner,
                <const (const void *) *>p_args,
                size,
                r_ret,
                r_error
            )
