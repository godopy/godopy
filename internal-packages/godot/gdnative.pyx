from .bindings cimport _cython_bindings

cdef public generic_gdnative_singleton():
    from importlib import import_module

    # Method bindings are not initialized yet
    _cython_bindings.ProjectSettings.__init_method_bindings()

    cdef _cython_bindings.ProjectSettings ps = _cython_bindings.ProjectSettings.get_singleton()
    gdlibrary_name = <object>ps.get_setting('python/config/gdnlib_module')
    gdlibrary = import_module(gdlibrary_name)

    if hasattr(gdlibrary, '_gdnative_singleton'):
        gdlibrary._gdnative_singleton()
