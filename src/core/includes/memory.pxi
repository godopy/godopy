cdef class _Memory:
    """
    Allocates, reallocates and frees memory.
    """
    cdef void *ptr
    cdef size_t num_bytes

    def __cinit__(self, size_t p_bytes) -> None:
        "Allocates memory."

        self.ptr = gdextension_interface_mem_alloc(p_bytes)
        self.num_bytes = p_bytes

        if self.ptr == NULL:
            raise MemoryError()

    cdef void *realloc(self, size_t p_bytes) except NULL nogil:
        "Reallocates memory."

        self.ptr = gdextension_interface_mem_realloc(self.ptr, p_bytes)

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

    def __dealloc__(self) -> None:
        "Frees memory."
        self.free()


    def as_array(self) -> np.ndarray:
        """
        Returns a numpy array that wraps the same memory buffer.
        """
        cdef uint8_t [:] view = <uint8_t[:self.num_bytes]>self.ptr

        return np.array(view, dtype=np.uint8_t, copy=False)
