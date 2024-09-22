
cdef object initialize_func = None
cdef object deinitialize_func = None


def set_global_functions():
    # printt = UtilityFunction('printt')
    # prints = UtilityFunction('prints')

    globals()['print'] = UtilityFunction('print')
    globals()['printerr'] = UtilityFunction('printerr')
    globals()['print_verbose'] = UtilityFunction('print_verbose')
    globals()['print_rich'] = UtilityFunction('print_rich')
    globals()['printraw'] = UtilityFunction('printraw')
    globals()['push_error'] = UtilityFunction('push_error')
    globals()['push_warningg'] = UtilityFunction('push_warning')


def initialize_level(ModuleInitializationLevel p_level):
    global initialize_func, deinitialize_func

    UtilityFunctions.print_verbose("GodoPy Python initialization started, level %d" % p_level)

    redirect_python_stdio()
    set_global_functions()

    try:
        import register_types
        initialize_func = getattr(register_types, 'initialize', None)  
        uninitialize_func = getattr(register_types, 'uninitialize', None)   
    except ImportError as exc:
        f = io.StringIO()
        traceback.print_exception(exc, file=f)
        exc_text = f.getvalue()
        if isinstance(exc, ModuleNotFoundError) and "'register_types'" in exc_text:
            UtilityFunctions.print_rich(
                "\n[color=orange]WARNING: 'register types' module was not found.[/color]\n"
            )
        else:
            UtilityFunctions.print_rich(
                "\n[color=red]ERROR: 'register types' module rased an exception:[/color]"
                "\n[color=orange]%s[/color]\n" % exc_text
            )

    if initialize_func is not None:
        initialize_func(p_level)


def deinitialize_level(ModuleInitializationLevel p_level):
    global deinitialize_func

    UtilityFunctions.print_verbose("GodoPy Python cleanup, level %d" % p_level)

    if deinitialize_func:
        deinitialize_func(p_level)

    return 0
