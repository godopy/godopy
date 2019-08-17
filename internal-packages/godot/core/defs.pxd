cdef enum Error:
    OK
    FAILED  # Generic fail error
    ERR_UNAVAILABLE  # What is requested is unsupported/unavailable
    ERR_UNCONFIGURED  # The object being used hasnt been properly set up yet
    ERR_UNAUTHORIZED  # Missing credentials for requested resource
    ERR_PARAMETER_RANGE_ERROR  # Parameter given out of range (5)
    ERR_OUT_OF_MEMORY  # Out of memory
    ERR_FILE_NOT_FOUND
    ERR_FILE_BAD_DRIVE
    ERR_FILE_BAD_PATH
    ERR_FILE_NO_PERMISSION  # (10)
    ERR_FILE_ALREADY_IN_USE
    ERR_FILE_CANT_OPEN
    ERR_FILE_CANT_WRITE
    ERR_FILE_CANT_READ
    ERR_FILE_UNRECOGNIZED  # (15)
    ERR_FILE_CORRUPT
    ERR_FILE_MISSING_DEPENDENCIES
    ERR_FILE_EOF
    ERR_CANT_OPEN  # Can't open a resource/socket/file
    ERR_CANT_CREATE  # (20)
    ERR_QUERY_FAILED  #
    ERR_ALREADY_IN_USE  #
    ERR_LOCKED  # resource is locked
    ERR_TIMEOUT
    ERR_CANT_CONNECT  # (25)
    ERR_CANT_RESOLVE
    ERR_CONNECTION_ERROR
    ERR_CANT_AQUIRE_RESOURCE
    ERR_CANT_FORK
    ERR_INVALID_DATA  # Data passed is invalid   (30)
    ERR_INVALID_PARAMETER  # Parameter passed is invalid
    ERR_ALREADY_EXISTS  # When adding, item already exists
    ERR_DOES_NOT_EXIST  # When retrieving/erasing, it item does not exist
    ERR_DATABASE_CANT_READ  # database is full
    ERR_DATABASE_CANT_WRITE  # database is full  (35)
    ERR_COMPILATION_FAILED
    ERR_METHOD_NOT_FOUND
    ERR_LINK_FAILED
    ERR_SCRIPT_FAILED
    ERR_CYCLIC_LINK
    ERR_INVALID_DECLARATION
    ERR_DUPLICATE_SYMBOL
    ERR_PARSE_ERROR
    ERR_BUSY
    ERR_SKIP  # (45)
    ERR_HELP  # user requested help!!
    ERR_BUG  # a bug in the software certainly happened, due to a double check failing or unexpected behavior.
    ERR_PRINTER_ON_FIRE  # the parallel port printer is engulfed in flames
    ERR_OMFG_THIS_IS_VERY_VERY_BAD  # shit happens, has never been used, though
    ERR_WTF = ERR_OMFG_THIS_IS_VERY_VERY_BAD  # short version of the above


cdef enum VariantType:
    VARIANT_NIL

    # Atomic types
    VARIANT_BOOL
    VARIANT_INT
    VARIANT_REAL
    VARIANT_STRING

    # Math types
    VARIANT_VECTOR2  # 5
    VARIANT_RECT2
    VARIANT_VECTOR3
    VARIANT_TRANSFORM2D
    VARIANT_PLANE
    VARIANT_QUAT  # 10
    VARIANT_RECT3
    VARIANT_BASIS
    VARIANT_TRANSFORM

    # Misc types
    VARIANT_COLOR
    VARIANT_NODE_PATH  # 15
    VARIANT__RID
    VARIANT_OBJECT
    VARIANT_DICTIONARY
    VARIANT_ARRAY

    # Arrays
    VARIANT_POOL_BYTE_ARRAY  # 20
    VARIANT_POOL_INT_ARRAY
    VARIANT_POOL_REAL_ARRAY
    VARIANT_POOL_STRING_ARRAY
    VARIANT_POOL_VECTOR2_ARRAY
    VARIANT_POOL_VECTOR3_ARRAY  # 25
    VARIANT_POOL_COLOR_ARRAY

    VARIANT_VARIANT_MAX


cdef enum VariantOperator:
    # Comparation
    VARIANT_OP_EQUAL
    VARIANT_OP_NOT_EQUAL
    VARIANT_OP_LESS
    VARIANT_OP_LESS_EQUAL
    VARIANT_OP_GREATER
    VARIANT_OP_GREATER_EQUAL

    # Mathematic
    VARIANT_OP_ADD
    VARIANT_OP_SUBSTRACT
    VARIANT_OP_MULTIPLY
    VARIANT_OP_DIVIDE
    VARIANT_OP_NEGATE
    VARIANT_OP_POSITIVE
    VARIANT_OP_MODULE
    VARIANT_OP_STRING_CONCAT

    # Bitwise
    VARIANT_OP_SHIFT_LEFT
    VARIANT_OP_SHIFT_RIGHT
    VARIANT_OP_BIT_AND
    VARIANT_OP_BIT_OR
    VARIANT_OP_BIT_XOR
    VARIANT_OP_BIT_NEGATE

    # Logic
    VARIANT_OP_AND
    VARIANT_OP_OR
    VARIANT_OP_XOR
    VARIANT_OP_NOT

    # Containment
    VARIANT_OP_IN
    VARIANT_OP_MAX


cdef enum Vector3Axis:
    VECTOR3_AXIS_X
    VECTOR3_AXIS_Y
    VECTOR3_AXIS_Z
