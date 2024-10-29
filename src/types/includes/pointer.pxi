cdef class Pointer:
    """
    Stores an opaque pointer.

    Intended for methods that pass or return any pointer values.
    """
    @staticmethod
    cdef Pointer create(const void *ptr):
        cdef Pointer self = Pointer.__new__(Pointer)
        self.ptr = <void *>ptr

        return self


cdef Pointer pointer_to_pyobject(const void *ptr):
    return Pointer.create(ptr)

cdef int pointer_from_pyobject(Pointer p_obj, void **r_ret) except -1:
    r_ret[0] = p_obj.ptr


cdef class Buffer:
    """
    Helper class that can make buffer Pointers useful
    """
    def __cinit__(self, Pointer pointer, int64_t size):
        self.ptr = <uint8_t *>pointer.ptr
        self.size = size

    def as_array(self, dtype=None):
        cdef uint8_t [:] view = <uint8_t[:self.size]>self.ptr

        return np.array(view, dtype=dtype, copy=False)


cdef class IntPointer(Pointer):
    """
    Helper class that can make integral Pointers useful
    """
    def __cinit__(self, Pointer other):
        self.ptr = other.ptr

    def set_value(self, int64_t value):
        if self.ptr != NULL:
            (<int64_t *>self.ptr)[0] = value
        else:
            raise ValueError("Could not unreference a NULL pointer")


cdef class FloatPointer(Pointer):
    """
    Helper class that can make numeric Pointers useful
    """
    def __cinit__(self, Pointer other):
        self.ptr = other.ptr

    def set_value(self, double value):
        if self.ptr != NULL:
            (<double *>self.ptr)[0] = value
        else:
            raise ValueError("Could not unreference a NULL pointer")
