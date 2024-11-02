# cdef GDExtensionBool _ext_set_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value) noexcept nogil:
#     if p_instance:
#         # TODO: set instance property
#         with gil:
#             return _extgil_set_bind(p_instance, p_name, p_value)

# cdef GDExtensionBool _extgil_set_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value) except -1:
#     cdef object wrapper = <object>p_instance
#     cdef str name = deref(<StringName *>p_name).py_str()
#     cdef object value = deref(<Variant *>p_value).pythonize()
#     print('SET BIND %r %s %r' % (wrapper, name, value))

#     return False


# cdef GDExtensionBool _ext_get_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) noexcept nogil:
#     if p_instance:
#         # TODO: get instance property
#         with gil:
#             return _extgil_get_bind(p_instance, p_name, r_ret)

# cdef GDExtensionBool _extgil_get_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) except -1:
#     cdef object wrapper = <object>p_instance
#     cdef str name = deref(<StringName *>p_name).py_str()
#     print('GET BIND %r %s' % (wrapper, name))

#     return False


# cdef GDExtensionPropertyInfo *_ext_get_property_list_bind(void *p_instance, uint32_t *r_count) noexcept nogil:
#     if r_count == NULL:
#         return NULL

#     cdef uint32_t count = deref(r_count)
#     with gil:
#         print('GETPROPLIST %x %d' % (<uint64_t>p_instance))
#     if not p_instance:
#         count = 0
#         return NULL
#     # TODO: Create and return property list
#     count = 0
#     return NULL


cdef void _ext_free_property_list_bind(void *p_instance, const GDExtensionPropertyInfo *p_list, uint32_t p_count) noexcept nogil:
    if p_instance:
        with gil:
            print('FREEPROPLIST %x' % (<uint64_t>p_instance))


cdef GDExtensionBool _ext_property_can_revert_bind(void *p_instance, GDExtensionConstStringNamePtr p_name) noexcept nogil:
    return False


cdef GDExtensionBool _ext_property_get_revert_bind(void *p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret) noexcept nogil:
    return False


cdef GDExtensionBool _ext_validate_property_bind(void *p_instance, GDExtensionPropertyInfo *p_property) noexcept nogil:
    return False


cdef void _ext_notification_bind(void *p_instance, int32_t p_what, GDExtensionBool p_reversed) noexcept nogil:
    if p_instance:
        with gil:
            _extgil_notification_bind(p_instance, p_what, p_reversed)

cdef int _extgil_notification_bind(void *p_instance, int32_t p_what, GDExtensionBool p_reversed) except -1:
    cdef object wrapper = <object>p_instance
    # print("NOTIFICATION %r %d %s" % (<uint64_t>p_instance, p_what, p_reversed))

    return 0


# cdef void _ext_to_string_bind(void *p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) noexcept nogil:
#     if p_instance:
#         with gil:
#             _extgil_to_string_bind(p_instance, r_is_valid, r_out)

# cdef int _extgil_to_string_bind(void *p_instance, GDExtensionBool *r_is_valid, GDExtensionStringPtr r_out) except -1:
#     cdef object wrapper = <object>p_instance
#     cdef str _repr = repr(wrapper)
#     print("TO_STRING %r %x" % (wrapper, <uint64_t>p_instance))
#     cdef GDExtensionBool is_valid = deref(r_is_valid)
#     cdef String out = deref(<String *>r_out)
#     is_valid = True
#     out = String(_repr)

#     return 0
