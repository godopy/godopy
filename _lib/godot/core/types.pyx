cimport cython

from libc.stdint cimport uint64_t, uint32_t, uint8_t, int64_t, int32_t
from libc.stddef cimport wchar_t

from godot_headers.gdnative_api cimport *

from ..globals cimport gdapi  #, PYGODOT_CHECK_NUMPY_API
from ._wrapped cimport _Wrapped, _PyWrapped
from . cimport cpp_types as cpp

from .tag_db cimport get_python_instance

from godopy cimport numpy as np

from cpython.ref cimport Py_INCREF
from cpython.object cimport PyObject, PyTypeObject, Py_LT, Py_EQ, Py_GT, Py_LE, Py_NE, Py_GE
from cython.operator cimport dereference as deref


def is_godot_wrapper_instance(object obj, object instances):
    return isinstance(obj, instances) and (<CoreTypeWrapper?>obj)._initialized

cdef class CoreTypeWrapper:
    def __cinit__(self):
        self._initialized = False


cdef class AABB(CoreTypeWrapper):
    @staticmethod
    cdef AABB from_cpp(cpp.AABB _cpp_object):
        cdef AABB self = AABB.__new__(AABB)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object pos=None, object size=None):
        if pos is not None and not is_godot_wrapper_instance(pos, Vector3):
           raise self._init_value_error(pos)
        if size is not None and not is_godot_wrapper_instance(pos, Vector3):
            raise self._init_value_error(size)

        if pos is not None and size is not None:
            self._cpp_object = cpp.AABB((<Vector3>pos)._cpp_object, (<Vector3>size)._cpp_object)
        else:
            self._cpp_object = cpp.AABB()
            if pos is not None:
                self._cpp_object.set_position((<Vector3>pos)._cpp_object)
            if size is not None:
                self._cpp_object.set_size((<Vector3>size)._cpp_object)

        self._initialized = True

    cdef cpp.AABB to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class Array(CoreTypeWrapper):
    @staticmethod
    cdef Array from_cpp(cpp.Array _cpp_object):
        cdef Array self = Array.__new__(Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    def make(*values):
        array = Array()

        for value in values:
            array._cpp_object.append(<const cpp.Variant &>value)

        return array

    def __init__(self, object other=None):
        if other is None:
            self._cpp_object = cpp.Array()
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.Array((<Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolByteArray):
            self._cpp_object = cpp.Array((<PoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolIntArray):
            self._cpp_object = cpp.Array((<PoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolRealArray):
            self._cpp_object = cpp.Array((<PoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolStringArray):
            self._cpp_object = cpp.Array((<PoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolVector2Array):
            self._cpp_object = cpp.Array((<PoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolVector3Array):
            self._cpp_object = cpp.Array((<PoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, PoolColorArray):
            self._cpp_object = cpp.Array((<PoolByteArray>other)._cpp_object)
        else:
            self._cpp_object = cpp.Array(other)

        self._initialized = True

    cdef cpp.Array to_cpp(self):
        self._internal_check()
        return self._cpp_object

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return <object>self._cpp_object[<const int>item]

    # def __setitem__(self, int item, object value):
    #     self._internal_check()
    #     self._cpp_object[<const int>item] = <const cpp.Variant &>value

    def to_tuple(self) -> tuple:
        return self._cpp_object.py_tuple()

    def to_list(self) -> list:
        return self._cpp_object.py_list()


cdef class Basis(CoreTypeWrapper):
    @staticmethod
    cdef Basis from_cpp(cpp.Basis _cpp_object):
        cdef Basis self = Basis.__new__(Basis)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None):
        self._cpp_object = cpp.Basis()
        self._initialized = True

    cdef cpp.Basis to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class Color(CoreTypeWrapper):
    @staticmethod
    cdef Color from_cpp(cpp.Color _cpp_object):
        cdef Color self = Color.__new__(Color)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.Color _cpp_object):
        cdef Color base = Color.from_cpp(_cpp_object)
        return base.to_numpy()

    @staticmethod
    def hex(self, uint32_t value):
        return Color.from_cpp(cpp.Color.hex(value))

    def __init__(self, float r=0, float g=0, float b=0, float a=1):
        self._cpp_object = cpp.Color(r, g, b, a)
        self._initialized = True

    cdef cpp.Color to_cpp(self):
        self._internal_check()
        return self._cpp_object

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return self._cpp_object[item]

    # def __setitem__(self, int item, float value):
    #     self._internal_check()
    #     self._cpp_object[item] = value

    def to_numpy(self) -> np.ndarray:
        cdef np.npy_intp *dims = [4]
        cdef float *ptr = <float *>&self._cpp_object

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, self)
        # print('BASE', np.get_array_base(arr))
        return arr

    def __repr__(self):
        self._internal_check()
        return 'Color(%s, %s, %s, %s)' % (self._cpp_object.r, self._cpp_object.g, self._cpp_object.b, self._cpp_object.a)


cdef class Dictionary(CoreTypeWrapper):
    @staticmethod
    cdef Dictionary from_cpp(cpp.Dictionary _cpp_object):
        cdef Dictionary self = Dictionary.__new__(Dictionary)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object other=None, **kwargs):
        if other is not None:
            self._cpp_object = cpp.Dictionary(other)
            # TODO: add kwargs
        elif kwargs:
            self._cpp_object = cpp.Dictionary(kwargs)
        else:
            self._cpp_object = cpp.Dictionary()
        self._initialized = True

    cdef cpp.Dictionary to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def to_dict(self) -> dict:
        return self._cpp_object.py_dict()


cdef class NodePath(CoreTypeWrapper):
    @staticmethod
    cdef NodePath from_cpp(cpp.NodePath _cpp_object):
        cdef NodePath self = NodePath.__new__(NodePath)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.NodePath()
        self._initialized = True

    cdef cpp.NodePath to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class Plane(CoreTypeWrapper):
    @staticmethod
    cdef Plane from_cpp(cpp.Plane _cpp_object):
        cdef Plane self = Plane.__new__(Plane)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Plane()
        self._initialized = True

    cdef cpp.Plane to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class PoolArray(CoreTypeWrapper):
    pass


@cython.no_gc_clear
cdef class PoolArrayReadAccess:
    pass


@cython.no_gc_clear
cdef class PoolArrayWriteAccess:
    pass


@cython.no_gc_clear
cdef class PoolByteArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolByteArray parent):
        cdef godot_pool_byte_array *arr = <godot_pool_byte_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_byte_array_read(arr)
        self._size = gdapi.godot_pool_byte_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_byte_array_read_access_destroy(self._read_access)

    def to_numpy(self, object base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef const uint8_t *ptr = gdapi.godot_pool_byte_array_read_access_ptr(self._read_access)

        cdef np.ndarray arr = np.array_new_simple_readonly(1, dims, np.NPY_UINT8, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


@cython.no_gc_clear
cdef class PoolByteArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolByteArray parent):
        cdef godot_pool_byte_array *arr = <godot_pool_byte_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_byte_array_write(arr)
        self._size = gdapi.godot_pool_byte_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_byte_array_write_access_destroy(self._write_access)

    def to_numpy(self, object base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef const uint8_t *ptr = gdapi.godot_pool_byte_array_write_access_ptr(self._write_access)

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_UINT8, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


cdef class PoolByteArray(PoolArray):
    @staticmethod
    cdef PoolByteArray from_cpp(cpp.PoolByteArray _cpp_object):
        cdef PoolByteArray self = PoolByteArray.__new__(PoolByteArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolByteArray _cpp_object, writable=False):
        cdef PoolByteArray base = PoolByteArray.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolByteArray _cpp_object):
        cdef PoolByteArray base = PoolByteArray.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolByteArray _cpp_object):
        cdef PoolByteArray base = PoolByteArray.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolByteArray:
        cdef PoolByteArray self = PoolByteArray.__new__(PoolByteArray)
        self._cpp_object = cpp.PoolByteArray(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolByteArray()
        elif is_godot_wrapper_instance(other, PoolByteArray):
            self._cpp_object = cpp.PoolByteArray((<PoolByteArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolByteArray((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolByteArray(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolByteArray(other)

        self._initialized = True

    cdef cpp.PoolByteArray to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolByteArrayReadAccess:
        return PoolByteArrayReadAccess(self)

    def write(self) -> PoolByteArrayWriteAccess:
        return PoolByteArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)



@cython.no_gc_clear
cdef class PoolIntArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolIntArray parent):
        cdef godot_pool_int_array *arr = <godot_pool_int_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_int_array_read(arr)
        self._size = gdapi.godot_pool_int_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_int_array_read_access_destroy(self._read_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef const int *ptr = gdapi.godot_pool_int_array_read_access_ptr(self._read_access)

        cdef np.ndarray arr = np.array_new_simple_readonly(1, dims, np.NPY_INT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


@cython.no_gc_clear
cdef class PoolIntArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolIntArray parent):
        cdef godot_pool_int_array *arr = <godot_pool_int_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_int_array_write(arr)
        self._size = gdapi.godot_pool_int_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_int_array_write_access_destroy(self._write_access)

    def to_numpy(self, object base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef int *ptr = gdapi.godot_pool_int_array_write_access_ptr(self._write_access)

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_INT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


cdef class PoolIntArray(PoolArray):
    @staticmethod
    cdef PoolIntArray from_cpp(cpp.PoolIntArray _cpp_object):
        cdef PoolIntArray self = PoolIntArray.__new__(PoolIntArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolIntArray _cpp_object, writable=False):
        cdef PoolIntArray base = PoolIntArray.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolIntArray _cpp_object):
        cdef PoolIntArray base = PoolIntArray.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolIntArray _cpp_object):
        cdef PoolIntArray base = PoolIntArray.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolIntArray:
        cdef PoolIntArray self = PoolIntArray.__new__(PoolIntArray)
        self._cpp_object = cpp.PoolIntArray(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolIntArray()
        elif is_godot_wrapper_instance(other, PoolIntArray):
            self._cpp_object = cpp.PoolIntArray((<PoolIntArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolIntArray((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolIntArray(<np.ndarray>other)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolIntArray(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolIntArray(other)

        self._initialized = True

    cdef cpp.PoolIntArray to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolIntArrayReadAccess:
        return PoolIntArrayReadAccess(self)

    def write(self) -> PoolIntArrayWriteAccess:
        return PoolIntArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)


@cython.no_gc_clear
cdef class PoolRealArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolRealArray parent):
        cdef godot_pool_real_array *arr = <godot_pool_real_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_real_array_read(arr)
        self._size = gdapi.godot_pool_real_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_real_array_read_access_destroy(self._read_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef const float *ptr = gdapi.godot_pool_real_array_read_access_ptr(self._read_access)

        cdef np.ndarray arr = np.array_new_simple_readonly(1, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


@cython.no_gc_clear
cdef class PoolRealArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolRealArray parent):
        cdef godot_pool_real_array *arr = <godot_pool_real_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_real_array_write(arr)
        self._size = gdapi.godot_pool_real_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_real_array_write_access_destroy(self._write_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef float *ptr = gdapi.godot_pool_real_array_write_access_ptr(self._write_access)

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


cdef class PoolRealArray(PoolArray):
    @staticmethod
    cdef PoolRealArray from_cpp(cpp.PoolRealArray _cpp_object):
        cdef PoolRealArray self = PoolRealArray.__new__(PoolRealArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolRealArray _cpp_object, writable=False):
        cdef PoolRealArray base = PoolRealArray.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolRealArray _cpp_object):
        cdef PoolRealArray base = PoolRealArray.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolRealArray _cpp_object):
        cdef PoolRealArray base = PoolRealArray.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolRealArray:
        cdef PoolRealArray self = PoolRealArray.__new__(PoolRealArray)
        self._cpp_object = cpp.PoolRealArray(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolRealArray()
        elif is_godot_wrapper_instance(other, PoolRealArray):
            self._cpp_object = cpp.PoolRealArray((<PoolRealArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolRealArray((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolRealArray(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolRealArray(other)

        self._initialized = True

    cdef cpp.PoolRealArray to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolRealArrayReadAccess:
        return PoolRealArrayReadAccess(self)

    def write(self) -> PoolRealArrayWriteAccess:
        return PoolRealArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)


@cython.no_gc_clear
cdef class PoolStringArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolStringArray parent):
        cdef godot_pool_string_array *arr = <godot_pool_string_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_string_array_read(arr)
        self._size = gdapi.godot_pool_string_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_string_array_read_access_destroy(self._read_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef const godot_string *ptr = gdapi.godot_pool_string_array_read_access_ptr(self._read_access)

        cdef np.npy_intp i
        cdef cpp.String *s
        cdef int itemsize = 1
        cdef int length

        for i in range(self._size):
            length = gdapi.godot_string_length(ptr + i)
            while length > itemsize:
                itemsize <<= 1

        cdef np.ndarray arr = np.PyArray_New(np.ndarray, 1, dims, np.NPY_UNICODE, NULL, NULL,
                                             itemsize * sizeof(wchar_t), np.NPY_ARRAY_CARRAY, <object>NULL)

        for i in range(self._size):
            s = <cpp.String *>(ptr + i)
            np.PyArray_SETITEM(arr, <char *>np.PyArray_GETPTR1(arr, i), s.py_str())

        np.PyArray_CLEARFLAGS(arr, np.NPY_ARRAY_WRITEABLE)

        return arr


@cython.no_gc_clear
cdef class PoolStringArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolStringArray parent):
        cdef godot_pool_string_array *arr = <godot_pool_string_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_string_array_write(arr)
        self._size = gdapi.godot_pool_string_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_string_array_write_access_destroy(self._write_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = &self._size;
        cdef godot_string *ptr = gdapi.godot_pool_string_array_write_access_ptr(self._write_access)

        cdef np.npy_intp i
        cdef cpp.String *s
        cdef int itemsize = 1
        cdef int length

        for i in range(self._size):
            length = gdapi.godot_string_length(ptr + i)
            while length > itemsize:
                itemsize <<= 1


        cdef np.ndarray arr = np.PyArray_New(np.ndarray, 1, dims, np.NPY_UNICODE, NULL, NULL,
                                             itemsize * sizeof(wchar_t), np.NPY_ARRAY_CARRAY, <object>NULL)

        for i in range(self._size):
            s = <cpp.String *>(ptr + i)
            np.PyArray_SETITEM(arr, <char *>np.PyArray_GETPTR1(arr, i), s.py_str())

        return arr


cdef class PoolStringArray(PoolArray):
    @staticmethod
    cdef PoolStringArray from_cpp(cpp.PoolStringArray _cpp_object):
        cdef PoolStringArray self = PoolStringArray.__new__(PoolStringArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolStringArray _cpp_object, writable=False):
        cdef PoolStringArray base = PoolStringArray.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolStringArray _cpp_object):
        cdef PoolStringArray base = PoolStringArray.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolStringArray _cpp_object):
        cdef PoolStringArray base = PoolStringArray.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolStringArray:
        cdef PoolStringArray self = PoolStringArray.__new__(PoolStringArray)
        self._cpp_object = cpp.PoolStringArray(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolStringArray()
        elif is_godot_wrapper_instance(other, PoolStringArray):
            self._cpp_object = cpp.PoolStringArray((<PoolStringArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolStringArray((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolStringArray(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolStringArray(other)

        self._initialized = True

    cdef cpp.PoolStringArray to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolStringArrayReadAccess:
        return PoolStringArrayReadAccess(self)

    def write(self) -> PoolStringArrayWriteAccess:
        return PoolStringArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)


@cython.no_gc_clear
cdef class PoolVector2ArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolVector2Array parent):
        cdef godot_pool_vector2_array *arr = <godot_pool_vector2_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_vector2_array_read(arr)
        self._size = gdapi.godot_pool_vector2_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_vector2_array_read_access_destroy(self._read_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = [self._size, 2];
        cdef const godot_vector2 *ptr = gdapi.godot_pool_vector2_array_read_access_ptr(self._read_access)

        cdef np.ndarray arr = np.array_new_simple_readonly(2, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


@cython.no_gc_clear
cdef class PoolVector2ArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolRealArray parent):
        cdef godot_pool_vector2_array *arr = <godot_pool_vector2_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_vector2_array_write(arr)
        self._size = gdapi.godot_pool_vector2_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_vector2_array_write_access_destroy(self._write_access)

    def to_numpy(self, object base=None) -> np.ndarray:
        cdef np.npy_intp *dims = [self._size, 2];
        cdef godot_vector2 *ptr = gdapi.godot_pool_vector2_array_write_access_ptr(self._write_access)

        cdef np.ndarray arr = np.array_new_simple(2, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


cdef class PoolVector2Array(PoolArray):
    @staticmethod
    cdef PoolVector2Array from_cpp(cpp.PoolVector2Array _cpp_object):
        cdef PoolVector2Array self = PoolVector2Array.__new__(PoolVector2Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolVector2Array _cpp_object, writable=False):
        cdef PoolVector2Array base = PoolVector2Array.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolVector2Array _cpp_object):
        cdef PoolVector2Array base = PoolVector2Array.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolVector2Array _cpp_object):
        cdef PoolVector2Array base = PoolVector2Array.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolVector2Array:
        cdef PoolVector2Array self = PoolVector2Array.__new__(PoolVector2Array)
        self._cpp_object = cpp.PoolVector2Array(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolVector2Array()
        elif is_godot_wrapper_instance(other, PoolVector2Array):
            self._cpp_object = cpp.PoolVector2Array((<PoolVector2Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolVector2Array((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolVector2Array(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolVector2Array(other)

        self._initialized = True

    cdef cpp.PoolVector2Array to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolVector2ArrayReadAccess:
        return PoolVector2ArrayReadAccess(self)

    def write(self) -> PoolVector2ArrayWriteAccess:
        return PoolVector2ArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)



@cython.no_gc_clear
cdef class PoolVector3ArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolVector3Array parent):
        cdef godot_pool_vector3_array *arr = <godot_pool_vector3_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_vector3_array_read(arr)
        self._size = gdapi.godot_pool_vector3_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_vector3_array_read_access_destroy(self._read_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = [self._size, 3];
        cdef const godot_vector3 *ptr = gdapi.godot_pool_vector3_array_read_access_ptr(self._read_access)

        cdef np.ndarray arr = np.array_new_simple_readonly(2, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


@cython.no_gc_clear
cdef class PoolVector3ArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolRealArray parent):
        cdef godot_pool_vector3_array *arr = <godot_pool_vector3_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_vector3_array_write(arr)
        self._size = gdapi.godot_pool_vector3_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_vector3_array_write_access_destroy(self._write_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = [self._size, 3];
        cdef godot_vector3 *ptr = gdapi.godot_pool_vector3_array_write_access_ptr(self._write_access)

        cdef np.ndarray arr = np.array_new_simple(2, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


cdef class PoolVector3Array(PoolArray):
    @staticmethod
    cdef PoolVector3Array from_cpp(cpp.PoolVector3Array _cpp_object):
        cdef PoolVector3Array self = PoolVector3Array.__new__(PoolVector3Array)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolVector3Array _cpp_object, writable=False):
        cdef PoolVector3Array base = PoolVector3Array.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolVector3Array _cpp_object):
        cdef PoolVector3Array base = PoolVector3Array.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolVector3Array _cpp_object):
        cdef PoolVector3Array base = PoolVector3Array.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolVector3Array:
        cdef PoolVector3Array self = PoolVector3Array.__new__(PoolVector3Array)
        self._cpp_object = cpp.PoolVector3Array(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolVector3Array()
        elif is_godot_wrapper_instance(other, PoolVector3Array):
            self._cpp_object = cpp.PoolVector3Array((<PoolVector3Array>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolVector3Array((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolVector3Array(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolVector3Array(other)

        self._initialized = True

    cdef cpp.PoolVector3Array to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolVector3ArrayReadAccess:
        return PoolVector3ArrayReadAccess(self)

    def write(self) -> PoolVector3ArrayWriteAccess:
        return PoolVector3ArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)



@cython.no_gc_clear
cdef class PoolColorArrayReadAccess(PoolArrayReadAccess):
    def __cinit__(self, PoolColorArray parent):
        cdef godot_pool_color_array *arr = <godot_pool_color_array *>&parent._cpp_object
        self._read_access = gdapi.godot_pool_color_array_read(arr)
        self._size = gdapi.godot_pool_color_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_color_array_read_access_destroy(self._read_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = [self._size, 4];
        cdef const godot_color *ptr = gdapi.godot_pool_color_array_read_access_ptr(self._read_access)

        cdef np.ndarray arr = np.array_new_simple_readonly(2, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


@cython.no_gc_clear
cdef class PoolColorArrayWriteAccess(PoolArrayWriteAccess):
    def __cinit__(self, PoolColorArray parent):
        cdef godot_pool_color_array *arr = <godot_pool_color_array *>&parent._cpp_object
        self._write_access = gdapi.godot_pool_color_array_write(arr)
        self._size = gdapi.godot_pool_color_array_size(arr)

    def __dealloc__(self):
        gdapi.godot_pool_color_array_write_access_destroy(self._write_access)

    def to_numpy(self, base=None) -> np.ndarray:
        cdef np.npy_intp *dims = [self._size, 4];
        cdef godot_color *ptr = gdapi.godot_pool_color_array_write_access_ptr(self._write_access)

        cdef np.ndarray arr = np.array_new_simple(2, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, base or self)
        # print('BASE', np.get_array_base(arr))
        return arr


cdef class PoolColorArray(PoolArray):
    @staticmethod
    cdef PoolColorArray from_cpp(cpp.PoolColorArray _cpp_object):
        cdef PoolColorArray self = PoolColorArray.__new__(PoolColorArray)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.PoolColorArray _cpp_object, writable=False):
        cdef PoolVector3Array base = PoolColorArray.from_cpp(_cpp_object)
        return base.to_numpy(writable=writable)

    @staticmethod
    cdef object from_cpp_to_pyreadaccess(cpp.PoolColorArray _cpp_object):
        cdef PoolColorArray base = PoolColorArray.from_cpp(_cpp_object)
        return base.read()

    @staticmethod
    cdef object from_cpp_to_pywriteaccess(cpp.PoolColorArray _cpp_object):
        cdef PoolColorArray base = PoolColorArray.from_cpp(_cpp_object)
        return base.write()

    @staticmethod
    def from_numpy(arr: np.ndarray) -> PoolColorArray:
        cdef PoolColorArray self = PoolColorArray.__new__(PoolColorArray)
        self._cpp_object = cpp.PoolColorArray(arr)
        self._initialized = True
        return self

    def __init__(self, other=None):
        if other is None:
            self._cpp_object = cpp.PoolColorArray()
        elif is_godot_wrapper_instance(other, PoolColorArray):
            self._cpp_object = cpp.PoolColorArray((<PoolColorArray>other)._cpp_object)
        elif is_godot_wrapper_instance(other, Array):
            self._cpp_object = cpp.PoolColorArray((<Array>other)._cpp_object)
        elif isinstance(other, np.ndarray):
            self._cpp_object = cpp.PoolColorArray(<np.ndarray>other)
        else:
            self._cpp_object = cpp.PoolColorArray(other)

        self._initialized = True

    cdef cpp.PoolColorArray to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def read(self) -> PoolColorArrayReadAccess:
        return PoolColorArrayReadAccess(self)

    def write(self) -> PoolColorArrayWriteAccess:
        return PoolColorArrayWriteAccess(self)

    def to_numpy(self, writable=False) -> np.ndarray:
        if writable:
            return self.write().to_numpy(base=self)
        else:
            return self.read().to_numpy(base=self)



cdef class Quat(CoreTypeWrapper):
    @staticmethod
    cdef Quat from_cpp(cpp.Quat _cpp_object):
        cdef Quat self = Quat.__new__(Quat)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Quat()
        self._initialized = True

    cdef cpp.Quat to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class Rect2(CoreTypeWrapper):
    @staticmethod
    cdef Rect2 from_cpp(cpp.Rect2 _cpp_object):
        cdef Rect2 self = Rect2.__new__(Rect2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, x=0, y=0, width=0, height=0):
        self._cpp_object = cpp.Rect2(x, y, width, height)
        self._initialized = True

    cdef cpp.Rect2 to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def to_numpy(self) -> np.ndarray:
        cdef np.npy_intp *dims = [4]
        cdef float *ptr = <float *>&self._cpp_object

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, self)
        # print('BASE', np.get_array_base(arr))
        return arr

    @property
    def position(self):
        self._internal_check()
        return Point2.from_cpp(self._cpp_object.position)

    @property
    def size(self):
        self._internal_check()
        return Size2.from_cpp(self._cpp_object.size)

    def __repr__(self):
        cdef Point2 pos = self.position
        cdef Size2 size = self.size
        return 'Rect2(%s, %s, %s, %s)' % (pos.x, pos.y, size.width, size.height)


cdef class RID(CoreTypeWrapper):
    @staticmethod
    cdef RID from_cpp(cpp.RID _cpp_object):
        cdef RID self = RID.__new__(RID)
        self._godot_rid = deref(<godot_rid *>&_cpp_object)
        self._initialized = True
        return self

    @staticmethod
    cdef RID from_godot_object(godot_object *_godot_object):
        cdef RID self = RID.__new__(RID)
        gdapi.godot_rid_new_with_resource(&self._godot_rid, _godot_object)
        self._initialized = True
        return self

    @staticmethod
    def from_wrapper_object(obj: _Wrapped) -> RID:
        cdef RID self = RID.__new__(RID)
        gdapi.godot_rid_new_with_resource(&self._godot_rid, (<_Wrapped>obj)._owner)
        self._initialized = True
        return self

    def __init__(self, object obj=None):
        cdef godot_object *_godot_object

        if obj is None:
            gdapi.godot_rid_new(&self._godot_rid)

        elif isinstance(obj, _Wrapped):
            _godot_object = (<_Wrapped>obj)._owner
            gdapi.godot_rid_new_with_resource(&self._godot_rid, _godot_object)

        else:
            self._init_value_error(obj)

        self._initialized = True

    def get_id(self) -> int32_t:
        return gdapi.godot_rid_get_id(&self._godot_rid)

    def __richcmp__(self, RID other, int op):
        if not is_godot_wrapper_instance(other, RID):
            self._argument_error(other)

        if op == Py_LT:
            return gdapi.godot_rid_operator_less(&self._godot_rid, &other._godot_rid)
        elif op == Py_EQ:
            return gdapi.godot_rid_operator_equal(&self._godot_rid, &other._godot_rid)
        elif op == Py_GT:
            return not gdapi.godot_rid_operator_less(&self._godot_rid, &other._godot_rid) and \
                   not gdapi.godot_rid_operator_equal(&self._godot_rid, &other._godot_rid)
        elif op == Py_LE:
            return gdapi.godot_rid_operator_less(&self._godot_rid, &other._godot_rid) or \
                   gdapi.godot_rid_operator_equal(&self._godot_rid, &other._godot_rid)
        elif op == Py_NE:
            return not gdapi.godot_rid_operator_equal(&self._godot_rid, &other._godot_rid)
        elif op == Py_GE:
            return not gdapi.godot_rid_operator_less(&self._godot_rid, &other._godot_rid)

    cdef cpp.RID to_cpp(self):
        self._internal_check()
        return deref(<cpp.RID *>&self._godot_rid)

    cdef godot_rid *to_godot_rid(self):
        self._internal_check()
        return &self._godot_rid


cdef class CharString(CoreTypeWrapper):
    @staticmethod
    cdef CharString from_cpp(cpp.CharString _cpp_object):
        cdef CharString self = CharString.__new__(CharString)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def length(self):
        self._internal_check()
        return self._cpp_object.length()

    def get_data(self):
        self._internal_check()
        return <bytes>self._cpp_object.get_data()

    cdef cpp.CharString to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class String(CoreTypeWrapper):
    @staticmethod
    cdef String from_cpp(cpp.String _cpp_object):
        cdef String self = String.__new__(String)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self, object content=None):
        if not content:
            # Initialize an empty String for all falsy values
            self._cpp_object = cpp.String()
        elif isinstance(content, basestring):
            self._cpp_object = cpp.String(content)
        else:
            self._cpp_object = cpp.String(str(content))

    cdef cpp.String to_cpp(self):
        self._internal_check()
        return self._cpp_object

    # @staticmethod
    # def num(double num, int decimals=-1):
    #     return String.from_cpp(String.num(num, decimals))

    # @staticmethod
    # def num_scientific(double num):
    #     return String.from_cpp(String.num_scientific(num))

    # @staticmethod
    # def num_real(double num):
    #     return String.from_cpp(String.num_real(num))

    # @staticmethod
    # def num_int64(int64_t num, int base=10, bint capitalize_hex=False):
    #     return String.from_cpp(String.num_int64(num, base, capitalize_hex))

    # TODO: chr, md5, hex_encode_buffer

    def __repr__(self):
        if self._initialized:
            return 'String(%r)' % self._cpp_object.py_str()
        return super().__repr__()

    def __str__(self):
        if self._initialized:
            return self._cpp_object.py_str()
        return '<nil>'

    # def __getitem__(self, int item):
    #     self._internal_check()
    #     return <Py_UNICODE>self._cpp_object[item]

    # TODO


cdef class Transform(CoreTypeWrapper):
    @staticmethod
    cdef Transform from_cpp(cpp.Transform _cpp_object):
        cdef Transform self = Transform.__new__(Transform)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Transform()
        self._initialized = True

    cdef cpp.Transform to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class Transform2D(CoreTypeWrapper):
    @staticmethod
    cdef Transform2D from_cpp(cpp.Transform2D _cpp_object):
        cdef Transform2D self = Transform2D.__new__(Transform2D)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __init__(self):
        self._cpp_object = cpp.Transform2D()
        self._initialized = True

    cdef cpp.Transform2D to_cpp(self):
        self._internal_check()
        return self._cpp_object


cdef class Vector2(CoreTypeWrapper):
    @staticmethod
    cdef Vector2 from_cpp(cpp.Vector2 _cpp_object):
        cdef Vector2 self = Vector2.__new__(Vector2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.Vector2 _cpp_object):
        cdef Vector2 base = Vector2.from_cpp(_cpp_object)
        return base.to_numpy()

    @staticmethod
    def from_numpy(np.ndarray arr):
        cdef Vector2 self = Vector2.__new__(Vector2)
        self._cpp_object = cpp.Vector2(arr)
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0):
        self._cpp_object = cpp.Vector2(x, y)
        self._initialized = True

    cdef cpp.Vector2 to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def to_numpy(self) -> np.ndarray:
        cdef np.npy_intp *dims = [2]
        cdef float *ptr = <float *>&self._cpp_object

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, self)
        # print('BASE', np.get_array_base(arr))
        return arr

    @property
    def x(self):
        self._internal_check()
        return self._cpp_object.x

    @property
    def y(self):
        self._internal_check()
        return self._cpp_object.y

    def __repr__(self):
        return 'Vector2(%s, %s)' % (self.x, self.y)


cdef class Point2(Vector2):
    @staticmethod
    cdef Point2 from_cpp(cpp.Point2 _cpp_object):
        cdef Point2 self = Point2.__new__(Point2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    def __repr__(self):
        return 'Point2(%s, %s)' % (self.x, self.y)


cdef class Size2(Vector2):
    @staticmethod
    cdef Size2 from_cpp(cpp.Size2 _cpp_object):
        cdef Size2 self = Size2.__new__(Size2)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @property
    def width(self):
        self._internal_check()
        return self._cpp_object.width

    @property
    def height(self):
        self._internal_check()
        return self._cpp_object.height

    def __repr__(self):
        return 'Size2(%s, %s)' % (self.width, self.height)


cdef class Vector3(CoreTypeWrapper):
    @staticmethod
    cdef Vector3 from_cpp(cpp.Vector3 _cpp_object):
        cdef Vector3 self = Vector3.__new__(Vector3)
        self._cpp_object = _cpp_object
        self._initialized = True
        return self

    @staticmethod
    cdef np.ndarray from_cpp_to_numpy(cpp.Vector3 _cpp_object):
        cdef Vector3 base = Vector3.from_cpp(_cpp_object)
        return base.to_numpy()

    @staticmethod
    def from_numpy(np.ndarray arr):
        cdef Vector3 self = Vector3.__new__(Vector3)
        self._cpp_object = cpp.Vector3(arr)
        self._initialized = True
        return self

    def __init__(self, float x=0, float y=0, float z=0):
        self._cpp_object = cpp.Vector3(x, y, z)
        self._initialized = True

    cdef cpp.Vector3 to_cpp(self):
        self._internal_check()
        return self._cpp_object

    def to_numpy(self) -> np.ndarray:
        cdef np.npy_intp *dims = [3]
        cdef float *ptr = <float *>&self._cpp_object

        cdef np.ndarray arr = np.array_new_simple(1, dims, np.NPY_FLOAT32, <void *>ptr)

        np.set_array_base(arr, self)
        # print('BASE', np.get_array_base(arr))
        return arr

    @property
    def x(self):
        self._internal_check()
        return self._cpp_object.x

    @property
    def y(self):
        self._internal_check()
        return self._cpp_object.y

    @property
    def z(self):
        self._internal_check()
        return self._cpp_object.z

    def __repr__(self):
        self._internal_check()
        return 'Vector3(%s, %s, %s)' % (self.x, self.y, self.z)


cdef public:
    # ctypedef AABB _python_aabb_wrapper
    # ctypedef Array _python_array_wrapper
    # ctypedef Basis _python_basis_wrapper
    # ctypedef Color _python_color_wrapper
    # ctypedef Dictionary _python_dictionary_wrapper
    # ctypedef NodePath _python_nodepath_wrapper
    # ctypedef Plane _python_plane_wrapper
    # ctypedef PoolByteArray _python_poolbytearray_wrapper
    # ctypedef PoolIntArray _python_poolintarray_wrapper
    # ctypedef PoolRealArray _python_poolrealarray_wrapper
    # ctypedef PoolStringArray _python_poolstringarray_wrapper
    # ctypedef PoolVector2Array _python_poolvector2array_wrapper
    # ctypedef PoolVector3Array _python_poolvector3array_wrapper
    # ctypedef PoolColorArray _python_poolcolorarray_wrapper
    # ctypedef Quat _python_quat_wrapper
    # ctypedef Rect2 _python_rect2_wrapper
    # ctypedef RID _python_rid_wrapper
    # ctypedef CharString _python_charstring_wrapper
    # ctypedef String _python_string_wrapper
    # ctypedef Transform _python_transform_wrapper
    # ctypedef Transform2D _python_transform2d_wrapper
    # ctypedef Vector2 _python_vector2_wrapper
    # ctypedef Vector3 _python_vector3_wrapper

    cdef type PyGodotWrapperType_AABB = AABB
    cdef type PyGodotWrapperType_Array = Array
    cdef type PyGodotWrapperType_Basis = Basis
    cdef type PyGodotWrapperType_Color = Color
    cdef type PyGodotWrapperType_Dictionary = Dictionary
    cdef type PyGodotWrapperType_NodePath = NodePath
    cdef type PyGodotWrapperType_Plane = Plane
    cdef type PyGodotWrapperType_PoolByteArray = PoolByteArray
    cdef type PyGodotWrapperType_PoolIntArray = PoolIntArray
    cdef type PyGodotWrapperType_PoolRealArray = PoolRealArray
    cdef type PyGodotWrapperType_PoolStringArray = PoolStringArray
    cdef type PyGodotWrapperType_PoolVector2Array = PoolVector2Array
    cdef type PyGodotWrapperType_PoolVector3Array = PoolVector3Array
    cdef type PyGodotWrapperType_PoolColorArray = PoolColorArray
    cdef type PyGodotWrapperType_Quat = Quat
    cdef type PyGodotWrapperType_Rect2 = Rect2
    cdef type PyGodotWrapperType_RID = RID
    cdef type PyGodotWrapperType_CharString = CharString
    cdef type PyGodotWrapperType_String = String
    cdef type PyGodotWrapperType_Transform = Transform
    cdef type PyGodotWrapperType_Transform2D = Transform2D
    cdef type PyGodotWrapperType_Vector2 = Vector2
    cdef type PyGodotWrapperType_Vector3 = Vector3
    cdef type PyGodotType__Wrapped = _Wrapped

    object _aabb_to_python_wrapper(cpp.AABB _obj):
        return <object>AABB.from_cpp(_obj)
    object _godot_array_to_python_wrapper(cpp.Array _obj):
        return <object>Array.from_cpp(_obj)
    object _basis_to_python_wrapper(cpp.Basis _obj):
        return <object>Basis.from_cpp(_obj)

    object _color_to_python_wrapper(cpp.Color _obj):
        return <object>Color.from_cpp(_obj)
    object _color_to_numpy(cpp.Color _obj):
        return <object>Color.from_cpp_to_numpy(_obj)

    object _godot_dictionary_to_python_wrapper(cpp.Dictionary  _obj):
        return <object>Dictionary.from_cpp(_obj)
    object _nodepath_to_python_wrapper(cpp.NodePath _obj):
        return <object>NodePath.from_cpp(_obj)
    object _plane_to_python_wrapper(cpp.Plane _obj):
        return <object>Plane.from_cpp(_obj)

    object _poolbytearray_to_python_wrapper(cpp.PoolByteArray _obj):
        return <object>PoolByteArray.from_cpp(_obj)
    object _poolbytearray_to_python_read(cpp.PoolByteArray _obj):
        return <object>PoolByteArray.from_cpp_to_pyreadaccess(_obj)
    object _poolbytearray_to_python_write(cpp.PoolByteArray _obj):
        return <object>PoolByteArray.from_cpp_to_pywriteaccess(_obj)
    object _poolbytearray_to_numpy(cpp.PoolByteArray _obj):
        return <object>PoolByteArray.from_cpp_to_numpy(_obj, writable=True)
    object _poolbytearray_to_numpy_ro(cpp.PoolByteArray _obj):
        return <object>PoolByteArray.from_cpp_to_numpy(_obj, writable=False)

    object _poolintarray_to_python_wrapper(cpp.PoolIntArray _obj):
        return <object>PoolIntArray.from_cpp(_obj)
    object _poolintarray_to_python_read(cpp.PoolIntArray _obj):
        return <object>PoolIntArray.from_cpp_to_pyreadaccess(_obj)
    object _poolintarray_to_python_write(cpp.PoolIntArray _obj):
        return <object>PoolIntArray.from_cpp_to_pywriteaccess(_obj)
    object _poolintarray_to_numpy(cpp.PoolIntArray _obj):
        return <object>PoolIntArray.from_cpp_to_numpy(_obj, writable=True)
    object _poolintarray_to_numpy_ro(cpp.PoolIntArray _obj):
        return <object>PoolIntArray.from_cpp_to_numpy(_obj, writable=False)

    object _poolrealarray_to_python_wrapper(cpp.PoolRealArray _obj):
        return <object>PoolRealArray.from_cpp(_obj)
    object _poolrealarray_to_python_read(cpp.PoolRealArray _obj):
        return <object>PoolRealArray.from_cpp_to_pyreadaccess(_obj)
    object _poolrealarray_to_python_write(cpp.PoolRealArray _obj):
        return <object>PoolRealArray.from_cpp_to_pywriteaccess(_obj)
    object _poolrealarray_to_numpy(cpp.PoolRealArray _obj):
        return <object>PoolRealArray.from_cpp_to_numpy(_obj, writable=True)
    object _poolrealarray_to_numpy_ro(cpp.PoolRealArray _obj):
        return <object>PoolRealArray.from_cpp_to_numpy(_obj, writable=False)

    object _poolstringarray_to_python_wrapper(cpp.PoolStringArray _obj):
        return <object>PoolStringArray.from_cpp(_obj)
    object _poolstringarray_to_python_read(cpp.PoolStringArray _obj):
        return <object>PoolStringArray.from_cpp_to_pyreadaccess(_obj)
    object _poolstringarray_to_python_write(cpp.PoolStringArray _obj):
        return <object>PoolStringArray.from_cpp_to_pywriteaccess(_obj)
    object _poolstringarray_to_numpy(cpp.PoolStringArray _obj):
        return <object>PoolStringArray.from_cpp_to_numpy(_obj, writable=True)
    object _poolstringarray_to_numpy_ro(cpp.PoolStringArray _obj):
        return <object>PoolStringArray.from_cpp_to_numpy(_obj, writable=False)

    object _poolvector2array_to_python_wrapper(cpp.PoolVector2Array _obj):
        return <object>PoolVector2Array.from_cpp(_obj)
    object _poolvector2array_to_python_read(cpp.PoolVector2Array _obj):
        return <object>PoolVector2Array.from_cpp_to_pyreadaccess(_obj)
    object _poolvector2array_to_python_write(cpp.PoolVector2Array _obj):
        return <object>PoolVector2Array.from_cpp_to_pywriteaccess(_obj)
    object _poolvector2array_to_numpy(cpp.PoolVector2Array _obj):
        return <object>PoolVector2Array.from_cpp_to_numpy(_obj, writable=True)
    object _poolvector2array_to_numpy_ro(cpp.PoolVector2Array _obj):
        return <object>PoolVector2Array.from_cpp_to_numpy(_obj, writable=False)

    object _poolvector3array_to_python_wrapper(cpp.PoolVector3Array _obj):
        return <object>PoolVector3Array.from_cpp(_obj)
    object _poolvector3array_to_python_read(cpp.PoolVector3Array _obj):
        return <object>PoolVector3Array.from_cpp_to_pyreadaccess(_obj)
    object _poolvector3array_to_python_write(cpp.PoolVector3Array _obj):
        return <object>PoolVector3Array.from_cpp_to_pywriteaccess(_obj)
    object _poolvector3array_to_numpy(cpp.PoolVector3Array _obj):
        return <object>PoolVector3Array.from_cpp_to_numpy(_obj, writable=True)
    object _poolvector3array_to_numpy_ro(cpp.PoolVector3Array _obj):
        return <object>PoolVector3Array.from_cpp_to_numpy(_obj, writable=False)

    object _poolcolorarray_to_python_wrapper(cpp.PoolColorArray _obj):
        return <object>PoolColorArray.from_cpp(_obj)
    object _poolcolorarray_to_python_read(cpp.PoolColorArray _obj):
        return <object>PoolColorArray.from_cpp_to_pyreadaccess(_obj)
    object _poolcolorarray_to_python_write(cpp.PoolColorArray _obj):
        return <object>PoolColorArray.from_cpp_to_pywriteaccess(_obj)
    object _poolcolorarray_to_numpy(cpp.PoolColorArray _obj):
        return <object>PoolColorArray.from_cpp_to_numpy(_obj, writable=True)
    object _poolcolorarray_to_numpy_ro(cpp.PoolColorArray _obj):
        return <object>PoolColorArray.from_cpp_to_numpy(_obj, writable=False)

    object _quat_to_python_wrapper(cpp.Quat _obj):
        return <object>Quat.from_cpp(_obj)
    object _rect2_to_python_wrapper(cpp.Rect2 _obj):
        return Rect2.from_cpp(_obj)
    object _rid_to_python_wrapper(cpp.RID _obj):
        return <object>RID.from_cpp(_obj)
    object _godot_string_to_python_wrapper(cpp.String _obj):
        return <object>String.from_cpp(_obj)
    object _transform_to_python_wrapper(cpp.Transform _obj):
        return <object>Transform.from_cpp(_obj)
    object _transform2d_to_python_wrapper(cpp.Transform2D _obj):
        return <object>Transform2D.from_cpp(_obj)

    object _vector2_to_python_wrapper(cpp.Vector2 _obj):
        return <object>Vector2.from_cpp(_obj)
    object _vector2_to_numpy(cpp.Vector2 _obj):
        return <object>Vector2.from_cpp_to_numpy(_obj)

    object _vector3_to_python_wrapper(cpp.Vector3 _obj):
        return <object>Vector3.from_cpp(_obj)
    object _vector3_to_numpy(cpp.Vector3 _obj):
        return <object>Vector3.from_cpp_to_numpy(_obj)


    object _godot_object_to_cython_binding(godot_object *_owner):
        return get_python_instance(_owner);

    object _godot_object_to_python_binding(godot_object *_owner):
        return get_python_instance(_owner);


    # Caller is responsible for type-checking in all
    # "*_binding_to_*" and "_python_wrapper_to_*" functions

    godot_object *_cython_binding_to_godot_object(object wrapped):
        return (<_Wrapped>wrapped)._owner
    godot_object *_python_binding_to_godot_object(object wrapped):
        return (<_PyWrapped>wrapped)._owner

    godot_aabb *_python_wrapper_to_aabb(object wrapper):
        return <godot_aabb *>&(<AABB>wrapper)._cpp_object
    godot_array *_python_wrapper_to_godot_array(object wrapper):
        return <godot_array *>&(<Array>wrapper)._cpp_object
    godot_basis *_python_wrapper_to_basis(object wrapper):
        return <godot_basis *>&(<Basis>wrapper)._cpp_object
    godot_color *_python_wrapper_to_color(object wrapper):
        return <godot_color *>&(<Color>wrapper)._cpp_object
    godot_dictionary *_python_wrapper_to_godot_dictionary(object wrapper):
        return <godot_dictionary *>&(<Dictionary>wrapper)._cpp_object
    godot_node_path *_python_wrapper_to_nodepath(object wrapper):
        return <godot_node_path *>&(<NodePath>wrapper)._cpp_object
    godot_plane *_python_wrapper_to_plane(object wrapper):
        return <godot_plane *>&(<Plane>wrapper)._cpp_object
    godot_pool_byte_array *_python_wrapper_to_poolbytearray(object wrapper):
        return <godot_pool_byte_array *>&(<PoolByteArray>wrapper)._cpp_object
    godot_pool_int_array *_python_wrapper_to_poolintarray(object wrapper):
        return <godot_pool_int_array *>&(<PoolIntArray>wrapper)._cpp_object
    godot_pool_real_array *_python_wrapper_to_poolrealarray(object wrapper):
        return <godot_pool_real_array *>&(<PoolRealArray>wrapper)._cpp_object
    godot_pool_string_array *_python_wrapper_to_poolstringarray(object wrapper):
        return <godot_pool_string_array *>&(<PoolStringArray>wrapper)._cpp_object
    godot_pool_vector2_array *_python_wrapper_to_poolvector2array(object wrapper):
        return <godot_pool_vector2_array *>&(<PoolVector2Array>wrapper)._cpp_object
    godot_pool_vector3_array *_python_wrapper_to_poolvector3array(object wrapper):
        return <godot_pool_vector3_array *>&(<PoolVector3Array>wrapper)._cpp_object
    godot_pool_color_array *_python_wrapper_to_poolcolorarray(object wrapper):
        return <godot_pool_color_array *>&(<PoolColorArray>wrapper)._cpp_object
    godot_quat *_python_wrapper_to_quat(object wrapper):
        return <godot_quat *>&(<Quat>wrapper)._cpp_object
    godot_rect2 *_python_wrapper_to_rect2(object wrapper):
        return <godot_rect2 *>&(<Rect2>wrapper)._cpp_object
    godot_rid *_python_wrapper_to_rid(object wrapper):
        return (<RID>wrapper).to_godot_rid()
    godot_string *_python_wrapper_to_godot_string(object wrapper):
        return <godot_string *>&(<String>wrapper)._cpp_object
    godot_transform *_python_wrapper_to_transform(object wrapper):
        return <godot_transform *>&(<Transform>wrapper)._cpp_object
    godot_transform2d *_python_wrapper_to_transform2d(object wrapper):
        return <godot_transform2d *>&(<Transform2D>wrapper)._cpp_object
    godot_vector2 *_python_wrapper_to_vector2(object wrapper):
        return <godot_vector2 *>&(<Vector2>wrapper)._cpp_object
    godot_vector3 *_python_wrapper_to_vector3(object wrapper):
        return <godot_vector3 *>&(<Vector3>wrapper)._cpp_object

