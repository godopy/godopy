from .bindings cimport _cython_bindings

cdef public generic_gdnative_singleton():
    from importlib import import_module

    # Method bindings are not initialized yet
    _cython_bindings.ProjectSettingsClass.__init_method_bindings()

    assert _cython_bindings.ProjectSettings is not None

    gdlibrary_name = <object>_cython_bindings.ProjectSettings.get_setting('python/config/gdnlib_module')
    gdlibrary = import_module(gdlibrary_name)

    if hasattr(gdlibrary, '_gdnative_singleton'):
        gdlibrary._gdnative_singleton()
