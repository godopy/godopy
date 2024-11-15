from gdextension_interface cimport *

ctypedef float real_t

include "includes/variant.pxi"

include "includes/string.pxi"

include "includes/vector2.pxi"
include "includes/vector2i.pxi"
include "includes/rect2.pxi"
include "includes/rect2i.pxi"
include "includes/vector3.pxi"
include "includes/vector3i.pxi"
include "includes/transform2d.pxi"
include "includes/vector4.pxi"
include "includes/vector4i.pxi"
include "includes/plane.pxi"
include "includes/quaternion.pxi"
include "includes/aabb.pxi"
include "includes/basis.pxi"
include "includes/transform3d.pxi"
include "includes/projection.pxi"

include "includes/color.pxi"
include "includes/string_name.pxi"
include "includes/node_path.pxi"
include "includes/rid.pxi"
include "includes/callable.pxi"
include "includes/signal.pxi"
include "includes/object.pxi"
include "includes/dictionary.pxi"
include "includes/array.pxi"

cdef extern from *:
    # FIXME: Normal indexing does not work
    # : Indexing 'Array' not supported for index type 'int64_t'
    # : Indexing 'const Dictionary &' not supported for index type 'Variant'
    """
    _FORCE_INLINE_ godot::Variant godot_array_get_item(const godot::Array &arr, const int64_t i) {
        return arr[i];
    }
    _FORCE_INLINE_ void godot_array_set_item(godot::Array &arr, const int64_t i, const godot::Variant &value) {
        arr[i] = value;
    }

    template <typename T>
    _FORCE_INLINE_ godot::Variant godot_typed_array_get_item(const godot::TypedArray<T> &arr, const int64_t i) {
        return arr[i];
    }
    template <typename T>
    _FORCE_INLINE_ void godot_typed_array_set_item(godot::TypedArray<T> &arr, const int64_t i, const godot::Variant &value) {
        arr[i] = value;
    }

    _FORCE_INLINE_ godot::Variant godot_dictionary_get_item(const godot::Dictionary &d, const godot::Variant &key) {
        return d[key];
    }
    _FORCE_INLINE_ void godot_dictionary_set_item(godot::Dictionary &d, const godot::Variant &key,
                                                  const godot::Variant &value) {
        d[key] = value;
    }
    """
    cdef Variant godot_array_get_item(const Array &, const int64_t)
    cdef void godot_array_set_item(const Array &, const int64_t, const Variant &)

    cdef Variant godot_typed_array_get_item[T](const TypedArray[T] &, const int64_t)
    cdef void godot_typed_array_set_item[T](const TypedArray[T] &, const int64_t, const Variant &)

    cdef Variant godot_typed_array_bool_get_item "godot_typed_array_get_item" (const TypedArrayBool &, const int64_t)
    cdef void godot_typed_array_bool_set_item "godot_typed_array_set_item" (const TypedArrayBool &, const int64_t, const Variant &)

    cdef Variant godot_dictionary_get_item(const Dictionary &, const Variant &)
    cdef void godot_dictionary_set_item(const Dictionary &, const Variant &, const Variant &)

include "includes/packed_byte_array.pxi"
include "includes/packed_int32_array.pxi"
include "includes/packed_int64_array.pxi"
include "includes/packed_float32_array.pxi"
include "includes/packed_float64_array.pxi"
include "includes/packed_string_array.pxi"
include "includes/packed_vector2_array.pxi"
include "includes/packed_vector3_array.pxi"
include "includes/packed_color_array.pxi"
include "includes/packed_vector4_array.pxi"

include "includes/classdb.pxi"
include "includes/engine.pxi"
include "includes/os.pxi"
include "includes/project_settings.pxi"

include "includes/utility_functions.pxi"

include "includes/structs.pxi"

include "includes/hashfuncs.pxi"
include "includes/global_constants.pxi"
