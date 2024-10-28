cdef dict _ERROR_TO_STR = {
	<int>GDEXTENSION_CALL_ERROR_INVALID_METHOD: "[GDExtensionCallError #%d] Invalid method",
	<int>GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT: "[GDExtensionCallError #%d] Invalid argument: expected a different variant type",
	<int>GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS: "[GDExtensionCallError #%d] Too many arguments: expected lower number of arguments",
	<int>GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS: "[GDExtensionCallError #%d] Too few arguments: expected higher number of arguments",
	<int>GDEXTENSION_CALL_ERROR_INSTANCE_IS_NULL: "[GDExtensionCallError #%d] Instance is NULL",
    <int>GDEXTENSION_CALL_ERROR_METHOD_NOT_CONST: "GDExtensionCallError #%d (GDEXTENSION_CALL_ERROR_METHOD_NOT_CONST)"
}


class EngineCallException(Exception):
    pass

class EngineVariantCallException(EngineCallException):
    pass

class GDExtensionCallException(EngineVariantCallException):
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


class EnginePtrCallException(EngineCallException):
    pass


class PythonPtrCallException(EngineCallException):
    pass
