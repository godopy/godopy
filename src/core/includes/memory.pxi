@cython.final
cdef class _Memory:
    """
    Allocates, reallocates and frees memory.
    """
    def __cinit__(self, size_t p_bytes) -> None:
        "Allocates memory."

        if p_bytes > 0:
            self.ptr = gdextension_interface_mem_alloc(p_bytes)

            if self.ptr == NULL:
                raise MemoryError()
        else:
            self.ptr = NULL

        self.num_bytes = p_bytes

    def __dealloc__(self) -> None:
        self.free()

    def __repr__(self):
        return "<%s._Memory at 0x%08X %dB>" % (self.__class__.__module__, <uint64_t>self.ptr, self.num_bytes)

    cdef void *realloc(self, size_t p_bytes) except NULL nogil:
        "Reallocates memory."

        if self.ptr != NULL:
            self.ptr = gdextension_interface_mem_realloc(self.ptr, p_bytes)
        else:
            self.ptr = gdextension_interface_mem_alloc(p_bytes)

        if self.ptr == NULL:
            with gil:
                raise MemoryError()

        self.num_bytes = p_bytes

        return self.ptr

    cdef void free(self) noexcept nogil:
        "Frees memory."

        if self.ptr != NULL:
            gdextension_interface_mem_free(self.ptr)

            self.ptr = NULL

    def as_array(self) -> np.ndarray:
        """
        Returns a numpy array that wraps the same memory buffer.
        """
        cdef uint8_t [:] view = <uint8_t[:self.num_bytes]>self.ptr

        return np.array(view, dtype=np.uint8_t, copy=False)
