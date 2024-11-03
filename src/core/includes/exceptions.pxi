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
