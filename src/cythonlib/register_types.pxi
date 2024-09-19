
cdef object init_func = None
cdef object register_func = None
cdef object unregister_func = None
cdef object terminate_func = None


cpdef public int initialize_types(ModuleInitializationLevel p_level) except -1:
    global init_func, register_func, unregister_func, terminate_func

    redirect_python_stdio()

    _print_verbose("GodoPy Python initialization started, level %d" % p_level)

    register_func = None
    try:
        import register_types
        init_func = getattr(register_types, 'initialize', None)  
        register_func = getattr(register_types, 'register', None)
        unregister_func = getattr(register_types, 'unregister', None)
        terminate_func = getattr(register_types, 'terminate', None)   
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

    if init_func:
        # TODO: Call with init level, do all levels
        init_func(p_level)

    if register_func:
        register_func(p_level)


cpdef public int terminate_types(ModuleInitializationLevel p_level) except -1:
    global unregister_func, terminate_func

    print_verbose("GodoPy Python cleanup, level %d" % p_level)

    if unregister_func:
        unregister_func(p_level)
    if terminate_func:
        # TODO: Call with init level
        terminate_func(p_level)

    return 0
