def _exc_info_from_exc(exc) -> Tuple[bytes, bytes, int]:
    f = io.StringIO()
    traceback.print_exception(exc, file=f)
    exc_text = f.getvalue()
    exc_lines = exc_text.splitlines()

    info_line_str = exc_lines[-2]

    if info_line_str.strip().startswith('^'):
        info_line_str = exc_lines[-4]

    info_line = [s.strip().strip(',') for s in info_line_str.split()]

    filename = info_line[1].encode('utf-8')
    function = info_line[-1].encode('utf-8')
    lineno = int(info_line[3])

    return filename, function, lineno


def print_error(exc, message=None, *, bint editor_notify=True):
    """
    Logs an error to Godot's built-in debugger and to the OS terminal.
    """
    cdef bytes descr = str(exc).encode('utf-8'), msg
    cdef bytes filename = b'<unknown>', function = b'<unknown>'
    cdef int32_t lineno = -1

    try:
        frame = inspect.currentframe()
        try:
            frameinfo = inspect.getframeinfo(frame.f_back)

            filename = frameinfo.filename.encode('utf-8')
            function = frameinfo.function.encode('utf-8')
            lineno = int(frameinfo.lineno)
        finally:
            del frame
    except ValueError:
        # Cython code, no call stack to inspect
        # Read values from the traceback text if an Exception object was passed
        if isinstance(exc, Exception):
            filename, function, lineno = _exc_info_from_exc(exc)


    if message is not None:
        msg = str(message).encode('utf-8')
        gdextension_interface_print_error_with_message(descr, msg, function, filename, lineno, editor_notify)
    else:
        gdextension_interface_print_error(descr, function, filename, lineno, editor_notify)


def print_error_with_traceback(exc, message=None, *, bint editor_notify=True):
    """
    Logs an error with a traceback (if available) to Godot's built-in debugger and to the OS terminal.
    """
    if isinstance(exc, Exception):
        f = io.StringIO()
        traceback.print_exception(exc, file=f)
        UtilityFunctions.print_rich("[color=purple]%s[/color]" % f.getvalue())

    print_error(exc, message, editor_notify=editor_notify)


def print_warning(warning, message=None, *, bint editor_notify=True):
    """
    Logs a warning to Godot's built-in debugger and to the OS terminal.
    """
    cdef bytes descr = str(warning).encode('utf-8'), msg
    cdef bytes filename = b'<unknown>', function = b'<unknown>'
    cdef int32_t lineno = -1

    try:
        frame = inspect.currentframe()
        try:
            frameinfo = inspect.getframeinfo(frame.f_back)

            filename = frameinfo.filename.encode('utf-8')
            function = frameinfo.function.encode('utf-8')
            lineno = int(frameinfo.lineno)
        finally:
            del frame
    except ValueError:
        # Cython code, no call stack to inspect
        # Read values from the traceback text if the warning was passed as an Exception object
        if isinstance(warning, Exception):
            filename, function, lineno = _exc_info_from_exc(warning)

    if message is not None:
        msg = str(message).encode('utf-8')
        gdextension_interface_print_warning_with_message(descr, msg, function, filename, lineno, editor_notify)
    else:
        gdextension_interface_print_warning(descr, function, filename, lineno, editor_notify)


def print_script_error(exc, message=None, *, bint editor_notify=True):
    """
    Logs a script error to Godot's built-in debugger and to the OS terminal.
    """
    cdef bytes descr = str(exc).encode('utf-8'), msg
    cdef bytes filename = b'<unknown>', function = b'<unknown>'
    cdef int32_t lineno = -1

    try:
        frame = inspect.currentframe()
        try:
            frameinfo = inspect.getframeinfo(frame.f_back)

            filename = frameinfo.filename.encode('utf-8')
            function = frameinfo.function.encode('utf-8')
            lineno = int(frameinfo.lineno)
        finally:
            del frame
    except ValueError:
        # Cython code, no call stack to inspect
        # Read values from the traceback text if an Exception object was passed
        if isinstance(exc, Exception):
            filename, function, lineno = _exc_info_from_exc(exc)

    if message is not None:
        msg = str(message).encode('utf-8')
        gdextension_interface_print_script_error_with_message(descr, msg, function, filename, lineno, editor_notify)
    else:
        gdextension_interface_print_script_error(descr, function, filename, lineno, editor_notify)


def print_script_error_with_traceback(exc, message=None, *, bint editor_notify=True):
    """
    Logs an error with a traceback (if available) to Godot's built-in debugger and to the OS terminal.
    """
    if isinstance(exc, Exception):
        f = io.StringIO()
        traceback.print_exception(exc, file=f)
        UtilityFunctions.print_rich("[color=gray]%s[/color]" % f.getvalue())

    print_script_error(exc, message, editor_notify=editor_notify)


cdef dict _ERROR_TO_STR = {
	CALL_ERROR_INVALID_METHOD: "[CallError #%d] Invalid method",
	CALL_ERROR_INVALID_ARGUMENT: "[CallError #%d] Invalid argument: expected a different variant type",
	CALL_ERROR_TOO_MANY_ARGUMENTS: "[CallError #%d] Too many arguments: expected lower number of arguments",
	CALL_ERROR_TOO_FEW_ARGUMENTS: "[CallError #%d] Too few arguments: expected higher number of arguments",
	CALL_ERROR_INSTANCE_IS_NULL: "[CallError #%d] Instance is NULL",
    CALL_ERROR_METHOD_NOT_CONST: "CallError #%d (CALL_ERROR_METHOD_NOT_CONST)"
}


class GDExtensionError(Exception):
    pass


class GDExtensionArgumentError(GDExtensionError):
    pass


class GDExtensionEngineCallError(GDExtensionError):
    pass

class GDExtensionngineVariantCallError(GDExtensionEngineCallError):
    pass

class GDExtensionCallException(GDExtensionngineVariantCallError):
    def __init__(self, msg, error_code):
        if not msg:
            msg = _ERROR_TO_STR.get(error_code, '')
            if not msg:
                msg = "GDExtensionCallError %d occured during Variant Engine call"
            msg = msg % error_code

        super().__init__(msg)
        self.error_code = error_code

    def __int__(self):
        return self.error_code


class GDExtensionEnginePtrCallError(GDExtensionEngineCallError):
    pass


class GDExtensionPythonPtrCallError(GDExtensionEngineCallError):
    pass
