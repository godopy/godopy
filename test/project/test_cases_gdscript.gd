extends Node

signal test_signal

var res = TestResource.new()

var m_bool: bool
var m_int: int
var m_float: float
var m_string: String
var m_vector2: Vector2
var m_vector2i: Vector2i
var m_rect2: Rect2
var m_rect2i: Rect2i
var m_vector3: Vector3
var m_vector3i: Vector3i
var m_transform2d: Transform2D
var m_vector4: Vector4
var m_vector4i: Vector4i
var m_plane: Plane
var m_quaternion: Quaternion
var m_aabb: AABB
var m_basis: Basis
var m_transform3d: Transform3D
var m_projection: Projection
var m_color: Color
var m_string_name: StringName
var m_node_path: NodePath
var m_rid: RID
var m_object: Object
var m_callable: Callable
var m_signal: Signal
var m_dictionary: Dictionary
var m_array: Array
var m_packed_byte_array: PackedByteArray
var m_packed_int32_array: PackedInt32Array
var m_packed_int64_array: PackedInt64Array
var m_packed_float32_array: PackedFloat32Array
var m_packed_float64_array: PackedFloat64Array
var m_packed_string_array: PackedStringArray
var m_packed_vector2_array: PackedVector2Array
var m_packed_vector3_array: PackedVector3Array
var m_packed_color_array: PackedColorArray
var m_packed_vector4_array: PackedVector4Array


func test_callable():
	return true


func get_resource():
	return res


func get_callable():
	return test_callable


func get_signal():
	return test_signal


func test_atomic_types_in(p_bool: bool, p_int: int, p_float: float, p_string: String):
	m_bool = p_bool
	m_int = p_int
	m_float = p_float
	m_string = p_string


func test_atomic_types_out():
	res.atomic_args(res.bool_ret(), res.int_ret(), res.float_ret(), res.string_ret())


func test_math_types_1_in(p_vector2: Vector2, p_vector2i: Vector2i, p_rect2: Rect2,
						  p_rect2i: Rect2i, p_vector3: Vector3, p_vector3i: Vector3i,
						  p_transform2d: Transform2D, p_vector4: Vector4, p_vector4i: Vector4i):
	m_vector2 = p_vector2
	m_vector2i = p_vector2i
	m_rect2 = p_rect2
	m_rect2i = p_rect2i
	m_vector3 = p_vector3
	m_vector3i = p_vector3i
	m_transform2d = p_transform2d
	m_vector4 = p_vector4
	m_vector4i = p_vector4i


func test_math_types_1_out():
	res.math_args_1(
		res.vector2_ret(), res.vector2i_ret(),
		res.rect2_ret(), res.rect2i_ret(),
		res.vector3_ret(), res.vector3i_ret(),
		res.transform2d_ret(),
		res.vector4_ret(), res.vector4i_ret()
	)


func test_math_types_2_in(p_plane: Plane, p_quaternion: Quaternion, p_aabb: AABB,
						  p_basis: Basis, p_transform3d: Transform3D,
						  p_projection: Projection, p_color: Color):
	m_plane = p_plane
	m_quaternion = p_quaternion
	m_aabb = p_aabb
	m_basis = p_basis
	m_transform3d = p_transform3d
	m_projection = p_projection
	m_color = p_color


func test_math_types_2_out():
	res.math_args_2(
		res.plane_ret(), res.quaternion_ret(), res.aabb_ret(), res.basis_ret(),
		res.transform3d_ret(), res.projection_ret(), res.color_ret()
	)

func test_misc_types_in(p_string_name: StringName, p_node_path: NodePath,
						p_rid: RID, p_object: Object, p_callable: Callable,
						p_signal: Signal, p_dictionary: Dictionary, p_array: Array):
	m_string_name = p_string_name
	m_node_path = p_node_path
	m_rid = p_rid
	m_object = p_object
	m_callable = p_callable
	m_signal = p_signal
	m_dictionary = p_dictionary
	m_array = p_array


func test_misc_types_out():
	res.misc_args(
		res.string_name_ret(),
		res.node_path_ret(),
		res.get_rid(),
		res.object_ret(),
		test_callable,
		test_signal,
		res.dictionary_ret(),
		res.array_ret()
	)


func test_packed_array_types_in(p_packed_byte_array: PackedByteArray,
								p_packed_int32_array: PackedInt32Array,
								p_packed_int64_array: PackedInt64Array,
								p_packed_float32_array: PackedFloat32Array,
								p_packed_float64_array: PackedFloat64Array,
								p_packed_string_array: PackedStringArray,
								p_packed_vector2_array: PackedVector2Array,
								p_packed_vector3_array: PackedVector3Array,
								p_packed_color_array: PackedColorArray,
								p_packed_vector4_array: PackedVector4Array):
	m_packed_byte_array = p_packed_byte_array
	m_packed_int32_array = p_packed_int32_array
	m_packed_int64_array = p_packed_int64_array
	m_packed_float32_array = p_packed_float32_array
	m_packed_float64_array = p_packed_float64_array
	m_packed_string_array = p_packed_string_array
	m_packed_vector2_array = p_packed_vector2_array
	m_packed_vector3_array = p_packed_vector3_array
	m_packed_color_array = p_packed_color_array
	m_packed_vector4_array = p_packed_vector4_array


func test_packed_array_types_out():
	res.packed_array_args(
		res.packed_byte_array_ret(),
		res.packed_int32_array_ret(),
		res.packed_int64_array_ret(),
		res.packed_float32_array_ret(),
		res.packed_float64_array_ret(),
		res.packed_string_array_ret(),
		res.packed_vector2_array_ret(),
		res.packed_vector3_array_ret(),
		res.packed_color_array_ret(),
		res.packed_vector4_array_ret()
	)
