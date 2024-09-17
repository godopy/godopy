from godot_cpp cimport *
from cpython cimport ref
from libcpp.unordered_map cimport unordered_map

include "shortcuts.pxi"

cdef dict _core_singletons = {}

def core_singleton(cls):
    global _core_singletons
    def getinstance():
        if cls not in _core_singletons:
            _core_singletons[cls] = cls()
        return _core_singletons[cls]
    return getinstance

cdef class __CoreSingletonBase:
    """\
    Base class for a subset of Godot Engine core classes
    These do not cover all available APIs, only essentials
    needed to implement everythin else
    """
    cdef void *_owner
    def __init__(self):
        raise RuntimeError("COre SIngleton classes are not instantiatable")


# cdef StringName class_name_OS = StringName('OS')
# cdef void *class_name_ptr_OS = class_name_OS._native_ptr()

cdef class _OS(__CoreSingletonBase):
    def __cinit__(self):
        self._owner = _gde_global_get_singleton(SN('OS').ptr())

    cpdef str read_string_from_stdin(self):
        cdef GDExtensionMethodBindPtr mb = _gde_classdb_get_method_bind(
            SN('OS').ptr(),
            SN('read_string_from_stdin').ptr(),
            2841200299,
        )
        cdef String gd_ret
        with nogil:
            _gde_object_method_bind_ptrcall(mb, self._owner, NULL, &gd_ret)
        return gd_ret.py_str()


cdef class _ClassDb(__CoreSingletonBase):
    def __cinit__(self):
        self._owner = _gde_global_get_singleton(SN('CLassDB').ptr())


# To be defined in Python code like this:
# @core_singleton
# class OS(_OS): pass
