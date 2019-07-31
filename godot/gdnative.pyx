from pygodot.utils cimport _init_dynamic_loading


cdef public generic_gdnative_singleton():
    if _init_dynamic_loading() != 0: return -1

    import gdlibrary

    if hasattr(gdlibrary, 'gdnative_singleton'):
        gdlibrary.gdnative_singleton()
