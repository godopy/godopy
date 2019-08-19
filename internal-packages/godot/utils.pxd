cdef inline allow_pure_python_imports(mod_name):
    import sys
    sys.modules.pop(mod_name)

cdef inline allow_cython_python_imports(mod_name):
    import sys
    sys.modules[mod_name] = sys.modules['__pygodot_internal__' + mod_name]
