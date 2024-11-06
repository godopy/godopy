from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stddef cimport wchar_t

cdef extern from "gdextension_interface.h" nogil:

    ctypedef uint32_t char32_t

    ctypedef uint16_t char16_t

    cdef enum _GDExtensionVariantType_e:
        GDEXTENSION_VARIANT_TYPE_NIL
        GDEXTENSION_VARIANT_TYPE_BOOL
        GDEXTENSION_VARIANT_TYPE_INT
        GDEXTENSION_VARIANT_TYPE_FLOAT
        GDEXTENSION_VARIANT_TYPE_STRING
        GDEXTENSION_VARIANT_TYPE_VECTOR2
        GDEXTENSION_VARIANT_TYPE_VECTOR2I
        GDEXTENSION_VARIANT_TYPE_RECT2
        GDEXTENSION_VARIANT_TYPE_RECT2I
        GDEXTENSION_VARIANT_TYPE_VECTOR3
        GDEXTENSION_VARIANT_TYPE_VECTOR3I
        GDEXTENSION_VARIANT_TYPE_TRANSFORM2D
        GDEXTENSION_VARIANT_TYPE_VECTOR4
        GDEXTENSION_VARIANT_TYPE_VECTOR4I
        GDEXTENSION_VARIANT_TYPE_PLANE
        GDEXTENSION_VARIANT_TYPE_QUATERNION
        GDEXTENSION_VARIANT_TYPE_AABB
        GDEXTENSION_VARIANT_TYPE_BASIS
        GDEXTENSION_VARIANT_TYPE_TRANSFORM3D
        GDEXTENSION_VARIANT_TYPE_PROJECTION
        GDEXTENSION_VARIANT_TYPE_COLOR
        GDEXTENSION_VARIANT_TYPE_STRING_NAME
        GDEXTENSION_VARIANT_TYPE_NODE_PATH
        GDEXTENSION_VARIANT_TYPE_RID
        GDEXTENSION_VARIANT_TYPE_OBJECT
        GDEXTENSION_VARIANT_TYPE_CALLABLE
        GDEXTENSION_VARIANT_TYPE_SIGNAL
        GDEXTENSION_VARIANT_TYPE_DICTIONARY
        GDEXTENSION_VARIANT_TYPE_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_BYTE_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_INT32_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_INT64_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT64_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_STRING_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR2_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR3_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_COLOR_ARRAY
        GDEXTENSION_VARIANT_TYPE_PACKED_VECTOR4_ARRAY
        GDEXTENSION_VARIANT_TYPE_VARIANT_MAX

    ctypedef _GDExtensionVariantType_e GDExtensionVariantType

    cdef enum _GDExtensionVariantOperator_e:
        GDEXTENSION_VARIANT_OP_EQUAL
        GDEXTENSION_VARIANT_OP_NOT_EQUAL
        GDEXTENSION_VARIANT_OP_LESS
        GDEXTENSION_VARIANT_OP_LESS_EQUAL
        GDEXTENSION_VARIANT_OP_GREATER
        GDEXTENSION_VARIANT_OP_GREATER_EQUAL
        GDEXTENSION_VARIANT_OP_ADD
        GDEXTENSION_VARIANT_OP_SUBTRACT
        GDEXTENSION_VARIANT_OP_MULTIPLY
        GDEXTENSION_VARIANT_OP_DIVIDE
        GDEXTENSION_VARIANT_OP_NEGATE
        GDEXTENSION_VARIANT_OP_POSITIVE
        GDEXTENSION_VARIANT_OP_MODULE
        GDEXTENSION_VARIANT_OP_POWER
        GDEXTENSION_VARIANT_OP_SHIFT_LEFT
        GDEXTENSION_VARIANT_OP_SHIFT_RIGHT
        GDEXTENSION_VARIANT_OP_BIT_AND
        GDEXTENSION_VARIANT_OP_BIT_OR
        GDEXTENSION_VARIANT_OP_BIT_XOR
        GDEXTENSION_VARIANT_OP_BIT_NEGATE
        GDEXTENSION_VARIANT_OP_AND
        GDEXTENSION_VARIANT_OP_OR
        GDEXTENSION_VARIANT_OP_XOR
        GDEXTENSION_VARIANT_OP_NOT
        GDEXTENSION_VARIANT_OP_IN
        GDEXTENSION_VARIANT_OP_MAX

    ctypedef _GDExtensionVariantOperator_e GDExtensionVariantOperator

    ctypedef void* GDExtensionVariantPtr

    ctypedef void* GDExtensionConstVariantPtr

    ctypedef void* GDExtensionUninitializedVariantPtr

    ctypedef void* GDExtensionStringNamePtr

    ctypedef void* GDExtensionConstStringNamePtr

    ctypedef void* GDExtensionUninitializedStringNamePtr

    ctypedef void* GDExtensionStringPtr

    ctypedef void* GDExtensionConstStringPtr

    ctypedef void* GDExtensionUninitializedStringPtr

    ctypedef void* GDExtensionObjectPtr

    ctypedef void* GDExtensionConstObjectPtr

    ctypedef void* GDExtensionUninitializedObjectPtr

    ctypedef void* GDExtensionTypePtr

    ctypedef void* GDExtensionConstTypePtr

    ctypedef void* GDExtensionUninitializedTypePtr

    ctypedef void* GDExtensionMethodBindPtr

    ctypedef int64_t GDExtensionInt

    ctypedef uint8_t GDExtensionBool

    ctypedef uint64_t GDObjectInstanceID

    ctypedef void* GDExtensionRefPtr

    ctypedef void* GDExtensionConstRefPtr

    cdef enum _GDExtensionCallErrorType_e:
        GDEXTENSION_CALL_OK
        GDEXTENSION_CALL_ERROR_INVALID_METHOD
        GDEXTENSION_CALL_ERROR_INVALID_ARGUMENT
        GDEXTENSION_CALL_ERROR_TOO_MANY_ARGUMENTS
        GDEXTENSION_CALL_ERROR_TOO_FEW_ARGUMENTS
        GDEXTENSION_CALL_ERROR_INSTANCE_IS_NULL
        GDEXTENSION_CALL_ERROR_METHOD_NOT_CONST

    ctypedef _GDExtensionCallErrorType_e GDExtensionCallErrorType

    cdef struct _GDExtensionCallError_s:
        GDExtensionCallErrorType error
        int32_t argument
        int32_t expected

    ctypedef _GDExtensionCallError_s GDExtensionCallError

    ctypedef void (*GDExtensionVariantFromTypeConstructorFunc)(GDExtensionUninitializedVariantPtr, GDExtensionTypePtr)

    ctypedef void (*GDExtensionTypeFromVariantConstructorFunc)(GDExtensionUninitializedTypePtr, GDExtensionVariantPtr)

    ctypedef void (*GDExtensionPtrOperatorEvaluator)(GDExtensionConstTypePtr p_left, GDExtensionConstTypePtr p_right, GDExtensionTypePtr r_result)

    ctypedef void (*GDExtensionPtrBuiltInMethod)(GDExtensionTypePtr p_base, GDExtensionConstTypePtr* p_args, GDExtensionTypePtr r_return, int p_argument_count)

    ctypedef void (*GDExtensionPtrConstructor)(GDExtensionUninitializedTypePtr p_base, GDExtensionConstTypePtr* p_args)

    ctypedef void (*GDExtensionPtrDestructor)(GDExtensionTypePtr p_base)

    ctypedef void (*GDExtensionPtrSetter)(GDExtensionTypePtr p_base, GDExtensionConstTypePtr p_value)

    ctypedef void (*GDExtensionPtrGetter)(GDExtensionConstTypePtr p_base, GDExtensionTypePtr r_value)

    ctypedef void (*GDExtensionPtrIndexedSetter)(GDExtensionTypePtr p_base, GDExtensionInt p_index, GDExtensionConstTypePtr p_value)

    ctypedef void (*GDExtensionPtrIndexedGetter)(GDExtensionConstTypePtr p_base, GDExtensionInt p_index, GDExtensionTypePtr r_value)

    ctypedef void (*GDExtensionPtrKeyedSetter)(GDExtensionTypePtr p_base, GDExtensionConstTypePtr p_key, GDExtensionConstTypePtr p_value)

    ctypedef void (*GDExtensionPtrKeyedGetter)(GDExtensionConstTypePtr p_base, GDExtensionConstTypePtr p_key, GDExtensionTypePtr r_value)

    ctypedef uint32_t (*GDExtensionPtrKeyedChecker)(GDExtensionConstVariantPtr p_base, GDExtensionConstVariantPtr p_key)

    ctypedef void (*GDExtensionPtrUtilityFunction)(GDExtensionTypePtr r_return, GDExtensionConstTypePtr* p_args, int p_argument_count)

    ctypedef GDExtensionObjectPtr (*GDExtensionClassConstructor)()

    ctypedef void* (*GDExtensionInstanceBindingCreateCallback)(void* p_token, void* p_instance)

    ctypedef void (*GDExtensionInstanceBindingFreeCallback)(void* p_token, void* p_instance, void* p_binding)

    ctypedef GDExtensionBool (*GDExtensionInstanceBindingReferenceCallback)(void* p_token, void* p_binding, GDExtensionBool p_reference)

    cdef struct _GDExtensionInstanceBindingCallbacks_s:
        GDExtensionInstanceBindingCreateCallback create_callback
        GDExtensionInstanceBindingFreeCallback free_callback
        GDExtensionInstanceBindingReferenceCallback reference_callback

    ctypedef _GDExtensionInstanceBindingCallbacks_s GDExtensionInstanceBindingCallbacks

    ctypedef void* GDExtensionClassInstancePtr

    ctypedef GDExtensionBool (*GDExtensionClassSet)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value)

    ctypedef GDExtensionBool (*GDExtensionClassGet)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret)

    ctypedef uint64_t (*GDExtensionClassGetRID)(GDExtensionClassInstancePtr p_instance)

    cdef struct _GDExtensionPropertyInfo_s:
        GDExtensionVariantType type
        GDExtensionStringNamePtr name
        GDExtensionStringNamePtr class_name
        uint32_t hint
        GDExtensionStringPtr hint_string
        uint32_t usage

    ctypedef _GDExtensionPropertyInfo_s GDExtensionPropertyInfo

    cdef struct _GDExtensionMethodInfo_s:
        GDExtensionStringNamePtr name
        GDExtensionPropertyInfo return_value
        uint32_t flags
        int32_t id
        uint32_t argument_count
        GDExtensionPropertyInfo* arguments
        uint32_t default_argument_count
        GDExtensionVariantPtr* default_arguments

    ctypedef _GDExtensionMethodInfo_s GDExtensionMethodInfo

    ctypedef GDExtensionPropertyInfo* (*GDExtensionClassGetPropertyList)(GDExtensionClassInstancePtr p_instance, uint32_t* r_count)

    ctypedef void (*GDExtensionClassFreePropertyList)(GDExtensionClassInstancePtr p_instance, GDExtensionPropertyInfo* p_list)

    ctypedef void (*GDExtensionClassFreePropertyList2)(GDExtensionClassInstancePtr p_instance, GDExtensionPropertyInfo* p_list, uint32_t p_count)

    ctypedef GDExtensionBool (*GDExtensionClassPropertyCanRevert)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name)

    ctypedef GDExtensionBool (*GDExtensionClassPropertyGetRevert)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret)

    ctypedef GDExtensionBool (*GDExtensionClassValidateProperty)(GDExtensionClassInstancePtr p_instance, GDExtensionPropertyInfo* p_property)

    ctypedef void (*GDExtensionClassNotification)(GDExtensionClassInstancePtr p_instance, int32_t p_what)

    ctypedef void (*GDExtensionClassNotification2)(GDExtensionClassInstancePtr p_instance, int32_t p_what, GDExtensionBool p_reversed)

    ctypedef void (*GDExtensionClassToString)(GDExtensionClassInstancePtr p_instance, GDExtensionBool* r_is_valid, GDExtensionStringPtr p_out)

    ctypedef void (*GDExtensionClassReference)(GDExtensionClassInstancePtr p_instance)

    ctypedef void (*GDExtensionClassUnreference)(GDExtensionClassInstancePtr p_instance)

    ctypedef void (*GDExtensionClassCallVirtual)(GDExtensionClassInstancePtr p_instance, GDExtensionConstTypePtr* p_args, GDExtensionTypePtr r_ret)

    ctypedef GDExtensionObjectPtr (*GDExtensionClassCreateInstance)(void* p_class_userdata)

    ctypedef GDExtensionObjectPtr (*GDExtensionClassCreateInstance2)(void* p_class_userdata, GDExtensionBool p_notify_postinitialize)

    ctypedef void (*GDExtensionClassFreeInstance)(void* p_class_userdata, GDExtensionClassInstancePtr p_instance)

    ctypedef GDExtensionClassInstancePtr (*GDExtensionClassRecreateInstance)(void* p_class_userdata, GDExtensionObjectPtr p_object)

    ctypedef GDExtensionClassCallVirtual (*GDExtensionClassGetVirtual)(void* p_class_userdata, GDExtensionConstStringNamePtr p_name)

    ctypedef void* (*GDExtensionClassGetVirtualCallData)(void* p_class_userdata, GDExtensionConstStringNamePtr p_name)

    ctypedef void (*GDExtensionClassCallVirtualWithData)(GDExtensionClassInstancePtr p_instance, GDExtensionConstStringNamePtr p_name, void* p_virtual_call_userdata, GDExtensionConstTypePtr* p_args, GDExtensionTypePtr r_ret)

    cdef struct _GDExtensionClassCreationInfo_s:
        GDExtensionBool is_virtual
        GDExtensionBool is_abstract
        GDExtensionClassSet set_func
        GDExtensionClassGet get_func
        GDExtensionClassGetPropertyList get_property_list_func
        GDExtensionClassFreePropertyList free_property_list_func
        GDExtensionClassPropertyCanRevert property_can_revert_func
        GDExtensionClassPropertyGetRevert property_get_revert_func
        GDExtensionClassNotification notification_func
        GDExtensionClassToString to_string_func
        GDExtensionClassReference reference_func
        GDExtensionClassUnreference unreference_func
        GDExtensionClassCreateInstance create_instance_func
        GDExtensionClassFreeInstance free_instance_func
        GDExtensionClassGetVirtual get_virtual_func
        GDExtensionClassGetRID get_rid_func
        void* class_userdata

    ctypedef _GDExtensionClassCreationInfo_s GDExtensionClassCreationInfo

    cdef struct _GDExtensionClassCreationInfo2_s:
        GDExtensionBool is_virtual
        GDExtensionBool is_abstract
        GDExtensionBool is_exposed
        GDExtensionClassSet set_func
        GDExtensionClassGet get_func
        GDExtensionClassGetPropertyList get_property_list_func
        GDExtensionClassFreePropertyList free_property_list_func
        GDExtensionClassPropertyCanRevert property_can_revert_func
        GDExtensionClassPropertyGetRevert property_get_revert_func
        GDExtensionClassValidateProperty validate_property_func
        GDExtensionClassNotification2 notification_func
        GDExtensionClassToString to_string_func
        GDExtensionClassReference reference_func
        GDExtensionClassUnreference unreference_func
        GDExtensionClassCreateInstance create_instance_func
        GDExtensionClassFreeInstance free_instance_func
        GDExtensionClassRecreateInstance recreate_instance_func
        GDExtensionClassGetVirtual get_virtual_func
        GDExtensionClassGetVirtualCallData get_virtual_call_data_func
        GDExtensionClassCallVirtualWithData call_virtual_with_data_func
        GDExtensionClassGetRID get_rid_func
        void* class_userdata

    ctypedef _GDExtensionClassCreationInfo2_s GDExtensionClassCreationInfo2

    cdef struct _GDExtensionClassCreationInfo3_s:
        GDExtensionBool is_virtual
        GDExtensionBool is_abstract
        GDExtensionBool is_exposed
        GDExtensionBool is_runtime
        GDExtensionClassSet set_func
        GDExtensionClassGet get_func
        GDExtensionClassGetPropertyList get_property_list_func
        GDExtensionClassFreePropertyList2 free_property_list_func
        GDExtensionClassPropertyCanRevert property_can_revert_func
        GDExtensionClassPropertyGetRevert property_get_revert_func
        GDExtensionClassValidateProperty validate_property_func
        GDExtensionClassNotification2 notification_func
        GDExtensionClassToString to_string_func
        GDExtensionClassReference reference_func
        GDExtensionClassUnreference unreference_func
        GDExtensionClassCreateInstance create_instance_func
        GDExtensionClassFreeInstance free_instance_func
        GDExtensionClassRecreateInstance recreate_instance_func
        GDExtensionClassGetVirtual get_virtual_func
        GDExtensionClassGetVirtualCallData get_virtual_call_data_func
        GDExtensionClassCallVirtualWithData call_virtual_with_data_func
        GDExtensionClassGetRID get_rid_func
        void* class_userdata

    ctypedef _GDExtensionClassCreationInfo3_s GDExtensionClassCreationInfo3

    cdef struct _GDExtensionClassCreationInfo4_s:
        GDExtensionBool is_virtual
        GDExtensionBool is_abstract
        GDExtensionBool is_exposed
        GDExtensionBool is_runtime
        GDExtensionClassSet set_func
        GDExtensionClassGet get_func
        GDExtensionClassGetPropertyList get_property_list_func
        GDExtensionClassFreePropertyList2 free_property_list_func
        GDExtensionClassPropertyCanRevert property_can_revert_func
        GDExtensionClassPropertyGetRevert property_get_revert_func
        GDExtensionClassValidateProperty validate_property_func
        GDExtensionClassNotification2 notification_func
        GDExtensionClassToString to_string_func
        GDExtensionClassReference reference_func
        GDExtensionClassUnreference unreference_func
        GDExtensionClassCreateInstance2 create_instance_func
        GDExtensionClassFreeInstance free_instance_func
        GDExtensionClassRecreateInstance recreate_instance_func
        GDExtensionClassGetVirtual get_virtual_func
        GDExtensionClassGetVirtualCallData get_virtual_call_data_func
        GDExtensionClassCallVirtualWithData call_virtual_with_data_func
        void* class_userdata

    ctypedef _GDExtensionClassCreationInfo4_s GDExtensionClassCreationInfo4

    ctypedef void* GDExtensionClassLibraryPtr

    cdef enum _GDExtensionClassMethodFlags_e:
        GDEXTENSION_METHOD_FLAG_NORMAL
        GDEXTENSION_METHOD_FLAG_EDITOR
        GDEXTENSION_METHOD_FLAG_CONST
        GDEXTENSION_METHOD_FLAG_VIRTUAL
        GDEXTENSION_METHOD_FLAG_VARARG
        GDEXTENSION_METHOD_FLAG_STATIC
        GDEXTENSION_METHOD_FLAGS_DEFAULT

    ctypedef _GDExtensionClassMethodFlags_e GDExtensionClassMethodFlags

    cdef enum _GDExtensionClassMethodArgumentMetadata_e:
        GDEXTENSION_METHOD_ARGUMENT_METADATA_NONE
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT8
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT16
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT32
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_INT64
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT8
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT16
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT32
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_UINT64
        GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_FLOAT
        GDEXTENSION_METHOD_ARGUMENT_METADATA_REAL_IS_DOUBLE
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_CHAR16
        GDEXTENSION_METHOD_ARGUMENT_METADATA_INT_IS_CHAR32

    ctypedef _GDExtensionClassMethodArgumentMetadata_e GDExtensionClassMethodArgumentMetadata

    ctypedef void (*GDExtensionClassMethodCall)(void* method_userdata, GDExtensionClassInstancePtr p_instance, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError* r_error)

    ctypedef void (*GDExtensionClassMethodValidatedCall)(void* method_userdata, GDExtensionClassInstancePtr p_instance, GDExtensionConstVariantPtr* p_args, GDExtensionVariantPtr r_return)

    ctypedef void (*GDExtensionClassMethodPtrCall)(void* method_userdata, GDExtensionClassInstancePtr p_instance, GDExtensionConstTypePtr* p_args, GDExtensionTypePtr r_ret)

    cdef struct _GDExtensionClassMethodInfo_s:
        GDExtensionStringNamePtr name
        void* method_userdata
        GDExtensionClassMethodCall call_func
        GDExtensionClassMethodPtrCall ptrcall_func
        uint32_t method_flags
        GDExtensionBool has_return_value
        GDExtensionPropertyInfo* return_value_info
        GDExtensionClassMethodArgumentMetadata return_value_metadata
        uint32_t argument_count
        GDExtensionPropertyInfo* arguments_info
        GDExtensionClassMethodArgumentMetadata* arguments_metadata
        uint32_t default_argument_count
        GDExtensionVariantPtr* default_arguments

    ctypedef _GDExtensionClassMethodInfo_s GDExtensionClassMethodInfo

    cdef struct _GDExtensionClassVirtualMethodInfo_s:
        GDExtensionStringNamePtr name
        uint32_t method_flags
        GDExtensionPropertyInfo return_value
        GDExtensionClassMethodArgumentMetadata return_value_metadata
        uint32_t argument_count
        GDExtensionPropertyInfo* arguments
        GDExtensionClassMethodArgumentMetadata* arguments_metadata

    ctypedef _GDExtensionClassVirtualMethodInfo_s GDExtensionClassVirtualMethodInfo

    ctypedef void (*GDExtensionCallableCustomCall)(void* callable_userdata, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError* r_error)

    ctypedef GDExtensionBool (*GDExtensionCallableCustomIsValid)(void* callable_userdata)

    ctypedef void (*GDExtensionCallableCustomFree)(void* callable_userdata)

    ctypedef uint32_t (*GDExtensionCallableCustomHash)(void* callable_userdata)

    ctypedef GDExtensionBool (*GDExtensionCallableCustomEqual)(void* callable_userdata_a, void* callable_userdata_b)

    ctypedef GDExtensionBool (*GDExtensionCallableCustomLessThan)(void* callable_userdata_a, void* callable_userdata_b)

    ctypedef void (*GDExtensionCallableCustomToString)(void* callable_userdata, GDExtensionBool* r_is_valid, GDExtensionStringPtr r_out)

    ctypedef GDExtensionInt (*GDExtensionCallableCustomGetArgumentCount)(void* callable_userdata, GDExtensionBool* r_is_valid)

    cdef struct _GDExtensionCallableCustomInfo_s:
        void* callable_userdata
        void* token
        GDObjectInstanceID object_id
        GDExtensionCallableCustomCall call_func
        GDExtensionCallableCustomIsValid is_valid_func
        GDExtensionCallableCustomFree free_func
        GDExtensionCallableCustomHash hash_func
        GDExtensionCallableCustomEqual equal_func
        GDExtensionCallableCustomLessThan less_than_func
        GDExtensionCallableCustomToString to_string_func

    ctypedef _GDExtensionCallableCustomInfo_s GDExtensionCallableCustomInfo

    cdef struct _GDExtensionCallableCustomInfo2_s:
        void* callable_userdata
        void* token
        GDObjectInstanceID object_id
        GDExtensionCallableCustomCall call_func
        GDExtensionCallableCustomIsValid is_valid_func
        GDExtensionCallableCustomFree free_func
        GDExtensionCallableCustomHash hash_func
        GDExtensionCallableCustomEqual equal_func
        GDExtensionCallableCustomLessThan less_than_func
        GDExtensionCallableCustomToString to_string_func
        GDExtensionCallableCustomGetArgumentCount get_argument_count_func

    ctypedef _GDExtensionCallableCustomInfo2_s GDExtensionCallableCustomInfo2

    ctypedef void* GDExtensionScriptInstanceDataPtr

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceSet)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value)

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceGet)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret)

    ctypedef GDExtensionPropertyInfo* (*GDExtensionScriptInstanceGetPropertyList)(GDExtensionScriptInstanceDataPtr p_instance, uint32_t* r_count)

    ctypedef void (*GDExtensionScriptInstanceFreePropertyList)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionPropertyInfo* p_list)

    ctypedef void (*GDExtensionScriptInstanceFreePropertyList2)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionPropertyInfo* p_list, uint32_t p_count)

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceGetClassCategory)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionPropertyInfo* p_class_category)

    ctypedef GDExtensionVariantType (*GDExtensionScriptInstanceGetPropertyType)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionBool* r_is_valid)

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceValidateProperty)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionPropertyInfo* p_property)

    ctypedef GDExtensionBool (*GDExtensionScriptInstancePropertyCanRevert)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name)

    ctypedef GDExtensionBool (*GDExtensionScriptInstancePropertyGetRevert)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionVariantPtr r_ret)

    ctypedef GDExtensionObjectPtr (*GDExtensionScriptInstanceGetOwner)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef void (*GDExtensionScriptInstancePropertyStateAdd)(GDExtensionConstStringNamePtr p_name, GDExtensionConstVariantPtr p_value, void* p_userdata)

    ctypedef void (*GDExtensionScriptInstanceGetPropertyState)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionScriptInstancePropertyStateAdd p_add_func, void* p_userdata)

    ctypedef GDExtensionMethodInfo* (*GDExtensionScriptInstanceGetMethodList)(GDExtensionScriptInstanceDataPtr p_instance, uint32_t* r_count)

    ctypedef void (*GDExtensionScriptInstanceFreeMethodList)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionMethodInfo* p_list)

    ctypedef void (*GDExtensionScriptInstanceFreeMethodList2)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionMethodInfo* p_list, uint32_t p_count)

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceHasMethod)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name)

    ctypedef GDExtensionInt (*GDExtensionScriptInstanceGetMethodArgumentCount)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionConstStringNamePtr p_name, GDExtensionBool* r_is_valid)

    ctypedef void (*GDExtensionScriptInstanceCall)(GDExtensionScriptInstanceDataPtr p_self, GDExtensionConstStringNamePtr p_method, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_argument_count, GDExtensionVariantPtr r_return, GDExtensionCallError* r_error)

    ctypedef void (*GDExtensionScriptInstanceNotification)(GDExtensionScriptInstanceDataPtr p_instance, int32_t p_what)

    ctypedef void (*GDExtensionScriptInstanceNotification2)(GDExtensionScriptInstanceDataPtr p_instance, int32_t p_what, GDExtensionBool p_reversed)

    ctypedef void (*GDExtensionScriptInstanceToString)(GDExtensionScriptInstanceDataPtr p_instance, GDExtensionBool* r_is_valid, GDExtensionStringPtr r_out)

    ctypedef void (*GDExtensionScriptInstanceRefCountIncremented)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceRefCountDecremented)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef GDExtensionObjectPtr (*GDExtensionScriptInstanceGetScript)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef GDExtensionBool (*GDExtensionScriptInstanceIsPlaceholder)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef void* GDExtensionScriptLanguagePtr

    ctypedef GDExtensionScriptLanguagePtr (*GDExtensionScriptInstanceGetLanguage)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef void (*GDExtensionScriptInstanceFree)(GDExtensionScriptInstanceDataPtr p_instance)

    ctypedef void* GDExtensionScriptInstancePtr

    cdef struct _GDExtensionScriptInstanceInfo_s:
        GDExtensionScriptInstanceSet set_func
        GDExtensionScriptInstanceGet get_func
        GDExtensionScriptInstanceGetPropertyList get_property_list_func
        GDExtensionScriptInstanceFreePropertyList free_property_list_func
        GDExtensionScriptInstancePropertyCanRevert property_can_revert_func
        GDExtensionScriptInstancePropertyGetRevert property_get_revert_func
        GDExtensionScriptInstanceGetOwner get_owner_func
        GDExtensionScriptInstanceGetPropertyState get_property_state_func
        GDExtensionScriptInstanceGetMethodList get_method_list_func
        GDExtensionScriptInstanceFreeMethodList free_method_list_func
        GDExtensionScriptInstanceGetPropertyType get_property_type_func
        GDExtensionScriptInstanceHasMethod has_method_func
        GDExtensionScriptInstanceCall call_func
        GDExtensionScriptInstanceNotification notification_func
        GDExtensionScriptInstanceToString to_string_func
        GDExtensionScriptInstanceRefCountIncremented refcount_incremented_func
        GDExtensionScriptInstanceRefCountDecremented refcount_decremented_func
        GDExtensionScriptInstanceGetScript get_script_func
        GDExtensionScriptInstanceIsPlaceholder is_placeholder_func
        GDExtensionScriptInstanceSet set_fallback_func
        GDExtensionScriptInstanceGet get_fallback_func
        GDExtensionScriptInstanceGetLanguage get_language_func
        GDExtensionScriptInstanceFree free_func

    ctypedef _GDExtensionScriptInstanceInfo_s GDExtensionScriptInstanceInfo

    cdef struct _GDExtensionScriptInstanceInfo2_s:
        GDExtensionScriptInstanceSet set_func
        GDExtensionScriptInstanceGet get_func
        GDExtensionScriptInstanceGetPropertyList get_property_list_func
        GDExtensionScriptInstanceFreePropertyList free_property_list_func
        GDExtensionScriptInstanceGetClassCategory get_class_category_func
        GDExtensionScriptInstancePropertyCanRevert property_can_revert_func
        GDExtensionScriptInstancePropertyGetRevert property_get_revert_func
        GDExtensionScriptInstanceGetOwner get_owner_func
        GDExtensionScriptInstanceGetPropertyState get_property_state_func
        GDExtensionScriptInstanceGetMethodList get_method_list_func
        GDExtensionScriptInstanceFreeMethodList free_method_list_func
        GDExtensionScriptInstanceGetPropertyType get_property_type_func
        GDExtensionScriptInstanceValidateProperty validate_property_func
        GDExtensionScriptInstanceHasMethod has_method_func
        GDExtensionScriptInstanceCall call_func
        GDExtensionScriptInstanceNotification2 notification_func
        GDExtensionScriptInstanceToString to_string_func
        GDExtensionScriptInstanceRefCountIncremented refcount_incremented_func
        GDExtensionScriptInstanceRefCountDecremented refcount_decremented_func
        GDExtensionScriptInstanceGetScript get_script_func
        GDExtensionScriptInstanceIsPlaceholder is_placeholder_func
        GDExtensionScriptInstanceSet set_fallback_func
        GDExtensionScriptInstanceGet get_fallback_func
        GDExtensionScriptInstanceGetLanguage get_language_func
        GDExtensionScriptInstanceFree free_func

    ctypedef _GDExtensionScriptInstanceInfo2_s GDExtensionScriptInstanceInfo2

    cdef struct _GDExtensionScriptInstanceInfo3_s:
        GDExtensionScriptInstanceSet set_func
        GDExtensionScriptInstanceGet get_func
        GDExtensionScriptInstanceGetPropertyList get_property_list_func
        GDExtensionScriptInstanceFreePropertyList2 free_property_list_func
        GDExtensionScriptInstanceGetClassCategory get_class_category_func
        GDExtensionScriptInstancePropertyCanRevert property_can_revert_func
        GDExtensionScriptInstancePropertyGetRevert property_get_revert_func
        GDExtensionScriptInstanceGetOwner get_owner_func
        GDExtensionScriptInstanceGetPropertyState get_property_state_func
        GDExtensionScriptInstanceGetMethodList get_method_list_func
        GDExtensionScriptInstanceFreeMethodList2 free_method_list_func
        GDExtensionScriptInstanceGetPropertyType get_property_type_func
        GDExtensionScriptInstanceValidateProperty validate_property_func
        GDExtensionScriptInstanceHasMethod has_method_func
        GDExtensionScriptInstanceGetMethodArgumentCount get_method_argument_count_func
        GDExtensionScriptInstanceCall call_func
        GDExtensionScriptInstanceNotification2 notification_func
        GDExtensionScriptInstanceToString to_string_func
        GDExtensionScriptInstanceRefCountIncremented refcount_incremented_func
        GDExtensionScriptInstanceRefCountDecremented refcount_decremented_func
        GDExtensionScriptInstanceGetScript get_script_func
        GDExtensionScriptInstanceIsPlaceholder is_placeholder_func
        GDExtensionScriptInstanceSet set_fallback_func
        GDExtensionScriptInstanceGet get_fallback_func
        GDExtensionScriptInstanceGetLanguage get_language_func
        GDExtensionScriptInstanceFree free_func

    ctypedef _GDExtensionScriptInstanceInfo3_s GDExtensionScriptInstanceInfo3

    cdef enum _GDExtensionInitializationLevel_e:
        GDEXTENSION_INITIALIZATION_CORE
        GDEXTENSION_INITIALIZATION_SERVERS
        GDEXTENSION_INITIALIZATION_SCENE
        GDEXTENSION_INITIALIZATION_EDITOR
        GDEXTENSION_MAX_INITIALIZATION_LEVEL

    ctypedef _GDExtensionInitializationLevel_e GDExtensionInitializationLevel

    ctypedef void (*_GDExtensionInitialization_GDExtensionInitialization_initialize_ft)(void* userdata, GDExtensionInitializationLevel p_level)

    ctypedef void (*_GDExtensionInitialization_GDExtensionInitialization_deinitialize_ft)(void* userdata, GDExtensionInitializationLevel p_level)

    cdef struct _GDExtensionInitialization_s:
        GDExtensionInitializationLevel minimum_initialization_level
        void* userdata
        _GDExtensionInitialization_GDExtensionInitialization_initialize_ft initialize
        _GDExtensionInitialization_GDExtensionInitialization_deinitialize_ft deinitialize

    ctypedef _GDExtensionInitialization_s GDExtensionInitialization

    ctypedef void (*GDExtensionInterfaceFunctionPtr)()

    ctypedef GDExtensionInterfaceFunctionPtr (*GDExtensionInterfaceGetProcAddress)(char* p_function_name)

    ctypedef GDExtensionBool (*GDExtensionInitializationFunction)(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization* r_initialization)

    cdef struct _GDExtensionGodotVersion_s:
        uint32_t major
        uint32_t minor
        uint32_t patch
        char* string

    ctypedef _GDExtensionGodotVersion_s GDExtensionGodotVersion

    ctypedef void (*GDExtensionInterfaceGetGodotVersion)(GDExtensionGodotVersion* r_godot_version)

    ctypedef void* (*GDExtensionInterfaceMemAlloc)(size_t p_bytes)

    ctypedef void* (*GDExtensionInterfaceMemRealloc)(void* p_ptr, size_t p_bytes)

    ctypedef void (*GDExtensionInterfaceMemFree)(void* p_ptr)

    ctypedef void (*GDExtensionInterfacePrintError)(char* p_description, char* p_function, char* p_file, int32_t p_line, GDExtensionBool p_editor_notify)

    ctypedef void (*GDExtensionInterfacePrintErrorWithMessage)(char* p_description, char* p_message, char* p_function, char* p_file, int32_t p_line, GDExtensionBool p_editor_notify)

    ctypedef void (*GDExtensionInterfacePrintWarning)(char* p_description, char* p_function, char* p_file, int32_t p_line, GDExtensionBool p_editor_notify)

    ctypedef void (*GDExtensionInterfacePrintWarningWithMessage)(char* p_description, char* p_message, char* p_function, char* p_file, int32_t p_line, GDExtensionBool p_editor_notify)

    ctypedef void (*GDExtensionInterfacePrintScriptError)(char* p_description, char* p_function, char* p_file, int32_t p_line, GDExtensionBool p_editor_notify)

    ctypedef void (*GDExtensionInterfacePrintScriptErrorWithMessage)(char* p_description, char* p_message, char* p_function, char* p_file, int32_t p_line, GDExtensionBool p_editor_notify)

    ctypedef uint64_t (*GDExtensionInterfaceGetNativeStructSize)(GDExtensionConstStringNamePtr p_name)

    ctypedef void (*GDExtensionInterfaceVariantNewCopy)(GDExtensionUninitializedVariantPtr r_dest, GDExtensionConstVariantPtr p_src)

    ctypedef void (*GDExtensionInterfaceVariantNewNil)(GDExtensionUninitializedVariantPtr r_dest)

    ctypedef void (*GDExtensionInterfaceVariantDestroy)(GDExtensionVariantPtr p_self)

    ctypedef void (*GDExtensionInterfaceVariantCall)(GDExtensionVariantPtr p_self, GDExtensionConstStringNamePtr p_method, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_argument_count, GDExtensionUninitializedVariantPtr r_return, GDExtensionCallError* r_error)

    ctypedef void (*GDExtensionInterfaceVariantCallStatic)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_method, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_argument_count, GDExtensionUninitializedVariantPtr r_return, GDExtensionCallError* r_error)

    ctypedef void (*GDExtensionInterfaceVariantEvaluate)(GDExtensionVariantOperator p_op, GDExtensionConstVariantPtr p_a, GDExtensionConstVariantPtr p_b, GDExtensionUninitializedVariantPtr r_return, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantSet)(GDExtensionVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionConstVariantPtr p_value, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantSetNamed)(GDExtensionVariantPtr p_self, GDExtensionConstStringNamePtr p_key, GDExtensionConstVariantPtr p_value, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantSetKeyed)(GDExtensionVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionConstVariantPtr p_value, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantSetIndexed)(GDExtensionVariantPtr p_self, GDExtensionInt p_index, GDExtensionConstVariantPtr p_value, GDExtensionBool* r_valid, GDExtensionBool* r_oob)

    ctypedef void (*GDExtensionInterfaceVariantGet)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionUninitializedVariantPtr r_ret, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantGetNamed)(GDExtensionConstVariantPtr p_self, GDExtensionConstStringNamePtr p_key, GDExtensionUninitializedVariantPtr r_ret, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantGetKeyed)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionUninitializedVariantPtr r_ret, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantGetIndexed)(GDExtensionConstVariantPtr p_self, GDExtensionInt p_index, GDExtensionUninitializedVariantPtr r_ret, GDExtensionBool* r_valid, GDExtensionBool* r_oob)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantIterInit)(GDExtensionConstVariantPtr p_self, GDExtensionUninitializedVariantPtr r_iter, GDExtensionBool* r_valid)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantIterNext)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_iter, GDExtensionBool* r_valid)

    ctypedef void (*GDExtensionInterfaceVariantIterGet)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_iter, GDExtensionUninitializedVariantPtr r_ret, GDExtensionBool* r_valid)

    ctypedef GDExtensionInt (*GDExtensionInterfaceVariantHash)(GDExtensionConstVariantPtr p_self)

    ctypedef GDExtensionInt (*GDExtensionInterfaceVariantRecursiveHash)(GDExtensionConstVariantPtr p_self, GDExtensionInt p_recursion_count)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantHashCompare)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_other)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantBooleanize)(GDExtensionConstVariantPtr p_self)

    ctypedef void (*GDExtensionInterfaceVariantDuplicate)(GDExtensionConstVariantPtr p_self, GDExtensionVariantPtr r_ret, GDExtensionBool p_deep)

    ctypedef void (*GDExtensionInterfaceVariantStringify)(GDExtensionConstVariantPtr p_self, GDExtensionStringPtr r_ret)

    ctypedef GDExtensionVariantType (*GDExtensionInterfaceVariantGetType)(GDExtensionConstVariantPtr p_self)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantHasMethod)(GDExtensionConstVariantPtr p_self, GDExtensionConstStringNamePtr p_method)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantHasMember)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_member)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantHasKey)(GDExtensionConstVariantPtr p_self, GDExtensionConstVariantPtr p_key, GDExtensionBool* r_valid)

    ctypedef GDObjectInstanceID (*GDExtensionInterfaceVariantGetObjectInstanceId)(GDExtensionConstVariantPtr p_self)

    ctypedef void (*GDExtensionInterfaceVariantGetTypeName)(GDExtensionVariantType p_type, GDExtensionUninitializedStringPtr r_name)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantCanConvert)(GDExtensionVariantType p_from, GDExtensionVariantType p_to)

    ctypedef GDExtensionBool (*GDExtensionInterfaceVariantCanConvertStrict)(GDExtensionVariantType p_from, GDExtensionVariantType p_to)

    ctypedef GDExtensionVariantFromTypeConstructorFunc (*GDExtensionInterfaceGetVariantFromTypeConstructor)(GDExtensionVariantType p_type)

    ctypedef GDExtensionTypeFromVariantConstructorFunc (*GDExtensionInterfaceGetVariantToTypeConstructor)(GDExtensionVariantType p_type)

    ctypedef GDExtensionPtrOperatorEvaluator (*GDExtensionInterfaceVariantGetPtrOperatorEvaluator)(GDExtensionVariantOperator p_operator, GDExtensionVariantType p_type_a, GDExtensionVariantType p_type_b)

    ctypedef GDExtensionPtrBuiltInMethod (*GDExtensionInterfaceVariantGetPtrBuiltinMethod)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_method, GDExtensionInt p_hash)

    ctypedef GDExtensionPtrConstructor (*GDExtensionInterfaceVariantGetPtrConstructor)(GDExtensionVariantType p_type, int32_t p_constructor)

    ctypedef GDExtensionPtrDestructor (*GDExtensionInterfaceVariantGetPtrDestructor)(GDExtensionVariantType p_type)

    ctypedef void (*GDExtensionInterfaceVariantConstruct)(GDExtensionVariantType p_type, GDExtensionUninitializedVariantPtr r_base, GDExtensionConstVariantPtr* p_args, int32_t p_argument_count, GDExtensionCallError* r_error)

    ctypedef GDExtensionPtrSetter (*GDExtensionInterfaceVariantGetPtrSetter)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_member)

    ctypedef GDExtensionPtrGetter (*GDExtensionInterfaceVariantGetPtrGetter)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_member)

    ctypedef GDExtensionPtrIndexedSetter (*GDExtensionInterfaceVariantGetPtrIndexedSetter)(GDExtensionVariantType p_type)

    ctypedef GDExtensionPtrIndexedGetter (*GDExtensionInterfaceVariantGetPtrIndexedGetter)(GDExtensionVariantType p_type)

    ctypedef GDExtensionPtrKeyedSetter (*GDExtensionInterfaceVariantGetPtrKeyedSetter)(GDExtensionVariantType p_type)

    ctypedef GDExtensionPtrKeyedGetter (*GDExtensionInterfaceVariantGetPtrKeyedGetter)(GDExtensionVariantType p_type)

    ctypedef GDExtensionPtrKeyedChecker (*GDExtensionInterfaceVariantGetPtrKeyedChecker)(GDExtensionVariantType p_type)

    ctypedef void (*GDExtensionInterfaceVariantGetConstantValue)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_constant, GDExtensionUninitializedVariantPtr r_ret)

    ctypedef GDExtensionPtrUtilityFunction (*GDExtensionInterfaceVariantGetPtrUtilityFunction)(GDExtensionConstStringNamePtr p_function, GDExtensionInt p_hash)

    ctypedef void (*GDExtensionInterfaceStringNewWithLatin1Chars)(GDExtensionUninitializedStringPtr r_dest, char* p_contents)

    ctypedef void (*GDExtensionInterfaceStringNewWithUtf8Chars)(GDExtensionUninitializedStringPtr r_dest, char* p_contents)

    ctypedef void (*GDExtensionInterfaceStringNewWithUtf16Chars)(GDExtensionUninitializedStringPtr r_dest, char16_t* p_contents)

    ctypedef void (*GDExtensionInterfaceStringNewWithUtf32Chars)(GDExtensionUninitializedStringPtr r_dest, char32_t* p_contents)

    ctypedef void (*GDExtensionInterfaceStringNewWithWideChars)(GDExtensionUninitializedStringPtr r_dest, wchar_t* p_contents)

    ctypedef void (*GDExtensionInterfaceStringNewWithLatin1CharsAndLen)(GDExtensionUninitializedStringPtr r_dest, char* p_contents, GDExtensionInt p_size)

    ctypedef void (*GDExtensionInterfaceStringNewWithUtf8CharsAndLen)(GDExtensionUninitializedStringPtr r_dest, char* p_contents, GDExtensionInt p_size)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringNewWithUtf8CharsAndLen2)(GDExtensionUninitializedStringPtr r_dest, char* p_contents, GDExtensionInt p_size)

    ctypedef void (*GDExtensionInterfaceStringNewWithUtf16CharsAndLen)(GDExtensionUninitializedStringPtr r_dest, char16_t* p_contents, GDExtensionInt p_char_count)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringNewWithUtf16CharsAndLen2)(GDExtensionUninitializedStringPtr r_dest, char16_t* p_contents, GDExtensionInt p_char_count, GDExtensionBool p_default_little_endian)

    ctypedef void (*GDExtensionInterfaceStringNewWithUtf32CharsAndLen)(GDExtensionUninitializedStringPtr r_dest, char32_t* p_contents, GDExtensionInt p_char_count)

    ctypedef void (*GDExtensionInterfaceStringNewWithWideCharsAndLen)(GDExtensionUninitializedStringPtr r_dest, wchar_t* p_contents, GDExtensionInt p_char_count)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringToLatin1Chars)(GDExtensionConstStringPtr p_self, char* r_text, GDExtensionInt p_max_write_length)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringToUtf8Chars)(GDExtensionConstStringPtr p_self, char* r_text, GDExtensionInt p_max_write_length)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringToUtf16Chars)(GDExtensionConstStringPtr p_self, char16_t* r_text, GDExtensionInt p_max_write_length)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringToUtf32Chars)(GDExtensionConstStringPtr p_self, char32_t* r_text, GDExtensionInt p_max_write_length)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringToWideChars)(GDExtensionConstStringPtr p_self, wchar_t* r_text, GDExtensionInt p_max_write_length)

    ctypedef char32_t* (*GDExtensionInterfaceStringOperatorIndex)(GDExtensionStringPtr p_self, GDExtensionInt p_index)

    ctypedef char32_t* (*GDExtensionInterfaceStringOperatorIndexConst)(GDExtensionConstStringPtr p_self, GDExtensionInt p_index)

    ctypedef void (*GDExtensionInterfaceStringOperatorPlusEqString)(GDExtensionStringPtr p_self, GDExtensionConstStringPtr p_b)

    ctypedef void (*GDExtensionInterfaceStringOperatorPlusEqChar)(GDExtensionStringPtr p_self, char32_t p_b)

    ctypedef void (*GDExtensionInterfaceStringOperatorPlusEqCstr)(GDExtensionStringPtr p_self, char* p_b)

    ctypedef void (*GDExtensionInterfaceStringOperatorPlusEqWcstr)(GDExtensionStringPtr p_self, wchar_t* p_b)

    ctypedef void (*GDExtensionInterfaceStringOperatorPlusEqC32str)(GDExtensionStringPtr p_self, char32_t* p_b)

    ctypedef GDExtensionInt (*GDExtensionInterfaceStringResize)(GDExtensionStringPtr p_self, GDExtensionInt p_resize)

    ctypedef void (*GDExtensionInterfaceStringNameNewWithLatin1Chars)(GDExtensionUninitializedStringNamePtr r_dest, char* p_contents, GDExtensionBool p_is_static)

    ctypedef void (*GDExtensionInterfaceStringNameNewWithUtf8Chars)(GDExtensionUninitializedStringNamePtr r_dest, char* p_contents)

    ctypedef void (*GDExtensionInterfaceStringNameNewWithUtf8CharsAndLen)(GDExtensionUninitializedStringNamePtr r_dest, char* p_contents, GDExtensionInt p_size)

    ctypedef GDExtensionInt (*GDExtensionInterfaceXmlParserOpenBuffer)(GDExtensionObjectPtr p_instance, uint8_t* p_buffer, size_t p_size)

    ctypedef void (*GDExtensionInterfaceFileAccessStoreBuffer)(GDExtensionObjectPtr p_instance, uint8_t* p_src, uint64_t p_length)

    ctypedef uint64_t (*GDExtensionInterfaceFileAccessGetBuffer)(GDExtensionConstObjectPtr p_instance, uint8_t* p_dst, uint64_t p_length)

    ctypedef uint8_t* (*GDExtensionInterfaceImagePtrw)(GDExtensionObjectPtr p_instance)

    ctypedef uint8_t* (*GDExtensionInterfaceImagePtr)(GDExtensionObjectPtr p_instance)

    ctypedef void (*_GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask_p_func_ft)(void*, uint32_t)

    ctypedef int64_t (*GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask)(GDExtensionObjectPtr p_instance, _GDExtensionInterfaceWorkerThreadPoolAddNativeGroupTask_p_func_ft p_func, void* p_userdata, int p_elements, int p_tasks, GDExtensionBool p_high_priority, GDExtensionConstStringPtr p_description)

    ctypedef void (*_GDExtensionInterfaceWorkerThreadPoolAddNativeTask_p_func_ft)(void*)

    ctypedef int64_t (*GDExtensionInterfaceWorkerThreadPoolAddNativeTask)(GDExtensionObjectPtr p_instance, _GDExtensionInterfaceWorkerThreadPoolAddNativeTask_p_func_ft p_func, void* p_userdata, GDExtensionBool p_high_priority, GDExtensionConstStringPtr p_description)

    ctypedef uint8_t* (*GDExtensionInterfacePackedByteArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef uint8_t* (*GDExtensionInterfacePackedByteArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef float* (*GDExtensionInterfacePackedFloat32ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef float* (*GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef double* (*GDExtensionInterfacePackedFloat64ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef double* (*GDExtensionInterfacePackedFloat64ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef int32_t* (*GDExtensionInterfacePackedInt32ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef int32_t* (*GDExtensionInterfacePackedInt32ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef int64_t* (*GDExtensionInterfacePackedInt64ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef int64_t* (*GDExtensionInterfacePackedInt64ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionStringPtr (*GDExtensionInterfacePackedStringArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionStringPtr (*GDExtensionInterfacePackedStringArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedVector2ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedVector2ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedVector3ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedVector3ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedVector4ArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedVector4ArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedColorArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionTypePtr (*GDExtensionInterfacePackedColorArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionVariantPtr (*GDExtensionInterfaceArrayOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionInt p_index)

    ctypedef GDExtensionVariantPtr (*GDExtensionInterfaceArrayOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)

    ctypedef void (*GDExtensionInterfaceArrayRef)(GDExtensionTypePtr p_self, GDExtensionConstTypePtr p_from)

    ctypedef void (*GDExtensionInterfaceArraySetTyped)(GDExtensionTypePtr p_self, GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstVariantPtr p_script)

    ctypedef GDExtensionVariantPtr (*GDExtensionInterfaceDictionaryOperatorIndex)(GDExtensionTypePtr p_self, GDExtensionConstVariantPtr p_key)

    ctypedef GDExtensionVariantPtr (*GDExtensionInterfaceDictionaryOperatorIndexConst)(GDExtensionConstTypePtr p_self, GDExtensionConstVariantPtr p_key)

    ctypedef void (*GDExtensionInterfaceDictionarySetTyped)(GDExtensionTypePtr p_self, GDExtensionVariantType p_key_type, GDExtensionConstStringNamePtr p_key_class_name, GDExtensionConstVariantPtr p_key_script, GDExtensionVariantType p_value_type, GDExtensionConstStringNamePtr p_value_class_name, GDExtensionConstVariantPtr p_value_script)

    ctypedef void (*GDExtensionInterfaceObjectMethodBindCall)(GDExtensionMethodBindPtr p_method_bind, GDExtensionObjectPtr p_instance, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_arg_count, GDExtensionUninitializedVariantPtr r_ret, GDExtensionCallError* r_error)

    ctypedef void (*GDExtensionInterfaceObjectMethodBindPtrcall)(GDExtensionMethodBindPtr p_method_bind, GDExtensionObjectPtr p_instance, GDExtensionConstTypePtr* p_args, GDExtensionTypePtr r_ret)

    ctypedef void (*GDExtensionInterfaceObjectDestroy)(GDExtensionObjectPtr p_o)

    ctypedef GDExtensionObjectPtr (*GDExtensionInterfaceGlobalGetSingleton)(GDExtensionConstStringNamePtr p_name)

    ctypedef void* (*GDExtensionInterfaceObjectGetInstanceBinding)(GDExtensionObjectPtr p_o, void* p_token, GDExtensionInstanceBindingCallbacks* p_callbacks)

    ctypedef void (*GDExtensionInterfaceObjectSetInstanceBinding)(GDExtensionObjectPtr p_o, void* p_token, void* p_binding, GDExtensionInstanceBindingCallbacks* p_callbacks)

    ctypedef void (*GDExtensionInterfaceObjectFreeInstanceBinding)(GDExtensionObjectPtr p_o, void* p_token)

    ctypedef void (*GDExtensionInterfaceObjectSetInstance)(GDExtensionObjectPtr p_o, GDExtensionConstStringNamePtr p_classname, GDExtensionClassInstancePtr p_instance)

    ctypedef GDExtensionBool (*GDExtensionInterfaceObjectGetClassName)(GDExtensionConstObjectPtr p_object, GDExtensionClassLibraryPtr p_library, GDExtensionUninitializedStringNamePtr r_class_name)

    ctypedef GDExtensionObjectPtr (*GDExtensionInterfaceObjectCastTo)(GDExtensionConstObjectPtr p_object, void* p_class_tag)

    ctypedef GDExtensionObjectPtr (*GDExtensionInterfaceObjectGetInstanceFromId)(GDObjectInstanceID p_instance_id)

    ctypedef GDObjectInstanceID (*GDExtensionInterfaceObjectGetInstanceId)(GDExtensionConstObjectPtr p_object)

    ctypedef GDExtensionBool (*GDExtensionInterfaceObjectHasScriptMethod)(GDExtensionConstObjectPtr p_object, GDExtensionConstStringNamePtr p_method)

    ctypedef void (*GDExtensionInterfaceObjectCallScriptMethod)(GDExtensionObjectPtr p_object, GDExtensionConstStringNamePtr p_method, GDExtensionConstVariantPtr* p_args, GDExtensionInt p_argument_count, GDExtensionUninitializedVariantPtr r_return, GDExtensionCallError* r_error)

    ctypedef GDExtensionObjectPtr (*GDExtensionInterfaceRefGetObject)(GDExtensionConstRefPtr p_ref)

    ctypedef void (*GDExtensionInterfaceRefSetObject)(GDExtensionRefPtr p_ref, GDExtensionObjectPtr p_object)

    ctypedef GDExtensionScriptInstancePtr (*GDExtensionInterfaceScriptInstanceCreate)(GDExtensionScriptInstanceInfo* p_info, GDExtensionScriptInstanceDataPtr p_instance_data)

    ctypedef GDExtensionScriptInstancePtr (*GDExtensionInterfaceScriptInstanceCreate2)(GDExtensionScriptInstanceInfo2* p_info, GDExtensionScriptInstanceDataPtr p_instance_data)

    ctypedef GDExtensionScriptInstancePtr (*GDExtensionInterfaceScriptInstanceCreate3)(GDExtensionScriptInstanceInfo3* p_info, GDExtensionScriptInstanceDataPtr p_instance_data)

    ctypedef GDExtensionScriptInstancePtr (*GDExtensionInterfacePlaceHolderScriptInstanceCreate)(GDExtensionObjectPtr p_language, GDExtensionObjectPtr p_script, GDExtensionObjectPtr p_owner)

    ctypedef void (*GDExtensionInterfacePlaceHolderScriptInstanceUpdate)(GDExtensionScriptInstancePtr p_placeholder, GDExtensionConstTypePtr p_properties, GDExtensionConstTypePtr p_values)

    ctypedef GDExtensionScriptInstanceDataPtr (*GDExtensionInterfaceObjectGetScriptInstance)(GDExtensionConstObjectPtr p_object, GDExtensionObjectPtr p_language)

    ctypedef void (*GDExtensionInterfaceCallableCustomCreate)(GDExtensionUninitializedTypePtr r_callable, GDExtensionCallableCustomInfo* p_callable_custom_info)

    ctypedef void (*GDExtensionInterfaceCallableCustomCreate2)(GDExtensionUninitializedTypePtr r_callable, GDExtensionCallableCustomInfo2* p_callable_custom_info)

    ctypedef void* (*GDExtensionInterfaceCallableCustomGetUserData)(GDExtensionConstTypePtr p_callable, void* p_token)

    ctypedef GDExtensionObjectPtr (*GDExtensionInterfaceClassdbConstructObject)(GDExtensionConstStringNamePtr p_classname)

    ctypedef GDExtensionObjectPtr (*GDExtensionInterfaceClassdbConstructObject2)(GDExtensionConstStringNamePtr p_classname)

    ctypedef GDExtensionMethodBindPtr (*GDExtensionInterfaceClassdbGetMethodBind)(GDExtensionConstStringNamePtr p_classname, GDExtensionConstStringNamePtr p_methodname, GDExtensionInt p_hash)

    ctypedef void* (*GDExtensionInterfaceClassdbGetClassTag)(GDExtensionConstStringNamePtr p_classname)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClass)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, GDExtensionClassCreationInfo* p_extension_funcs)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClass2)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, GDExtensionClassCreationInfo2* p_extension_funcs)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClass3)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, GDExtensionClassCreationInfo3* p_extension_funcs)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClass4)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, GDExtensionClassCreationInfo4* p_extension_funcs)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassMethod)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionClassMethodInfo* p_method_info)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassVirtualMethod)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionClassVirtualMethodInfo* p_method_info)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassIntegerConstant)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_enum_name, GDExtensionConstStringNamePtr p_constant_name, GDExtensionInt p_constant_value, GDExtensionBool p_is_bitfield)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassProperty)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionPropertyInfo* p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassPropertyIndexed)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionPropertyInfo* p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter, GDExtensionInt p_index)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassPropertyGroup)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_group_name, GDExtensionConstStringPtr p_prefix)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassPropertySubgroup)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_subgroup_name, GDExtensionConstStringPtr p_prefix)

    ctypedef void (*GDExtensionInterfaceClassdbRegisterExtensionClassSignal)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_signal_name, GDExtensionPropertyInfo* p_argument_info, GDExtensionInt p_argument_count)

    ctypedef void (*GDExtensionInterfaceClassdbUnregisterExtensionClass)(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name)

    ctypedef void (*GDExtensionInterfaceGetLibraryPath)(GDExtensionClassLibraryPtr p_library, GDExtensionUninitializedStringPtr r_path)

    ctypedef void (*GDExtensionInterfaceEditorAddPlugin)(GDExtensionConstStringNamePtr p_class_name)

    ctypedef void (*GDExtensionInterfaceEditorRemovePlugin)(GDExtensionConstStringNamePtr p_class_name)

    ctypedef void (*GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8Chars)(char* p_data)

    ctypedef void (*GDExtensionsInterfaceEditorHelpLoadXmlFromUtf8CharsAndLen)(char* p_data, GDExtensionInt p_size)
