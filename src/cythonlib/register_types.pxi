
cdef object initialize_func = None
cdef object uninitialize_func = None


# cdef public int initialize_python_types(ModuleInitializationLevel p_level) except -1 nogil:
#     with gil:
#         return initialize_godopy_types(p_level)


# cdef public int uninitialize_python_types(ModuleInitializationLevel p_level) except -1 nogil:
#     with gil:
#         return uninitialize_godopy_types(p_level)


cpdef int initialize_godopy_types(ModuleInitializationLevel p_level) except -1:
    global initialize_func, uninitialize_func

    redirect_python_stdio()

    _print_verbose("GodoPy Python initialization started, level %d" % p_level)

    try:
        import register_types
        initialize_func = getattr(register_types, 'initialize', None)  
        uninitialize_func = getattr(register_types, 'uninitialize', None)   
    except ImportError as exc:
        f = io.StringIO()
        traceback.print_exception(exc, file=f)
        exc_text = f.getvalue()
        if isinstance(exc, ModuleNotFoundError) and "'register_types'" in exc_text:
            _print_rich(
                "\n[color=orange]WARNING: 'register types' module was not found.[/color]\n"
            )
        else:
            _print_rich(
                "\n[color=red]ERROR: 'register types' module rased an exception:[/color]"
                "\n[color=orange]%s[/color]\n" % exc_text
            )

    if initialize_func is not None:
        initialize_func(p_level)


cpdef int uninitialize_godopy_types(ModuleInitializationLevel p_level) except -1:
    global uninitialize_func

    print_verbose("GodoPy Python cleanup, level %d" % p_level)

    if uninitialize_func:
        uninitialize_func(p_level)

    return 0
