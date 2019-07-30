from pygodot.utils cimport _init_dynamic_loading


cdef public int _generic_pygodot_gdnative_singleton() except -1:
    if _init_dynamic_loading() != 0: return -1

    import gdlibrary

    if hasattr(gdlibrary, 'gdnative_singleton'):
        gdlibrary.gdnative_singleton()
