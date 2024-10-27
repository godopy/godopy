extends Node

signal test_signal

var res = TestResource.new()

var m_bool: bool
var m_int: int
var m_float: float
var m_string: String

func get_resource():
	return res

func test_atomic_types_out():
	res.atomic_args(res.bool_ret(), res.int_ret(), res.float_ret(), res.string_ret())


func test_atomic_types_in_test(p_bool: bool):
	m_bool = p_bool

func test_atomic_types_in(p_bool: bool, p_int: int, p_float: float, p_string: String):
	m_bool = p_bool
	m_int = p_int
	m_float = p_float
	m_string = p_string


func test_math_types_1_out_1():
	res.math_args_1(
		Vector2(2.5, 5), Vector2i(5, 10),
		Rect2(0, 1, 100, 200), Rect2i(0, 1, 100, 200),
		Vector3(2.5, 5, 10), Vector3i(5, 10, 20),
		Transform2D(Vector2(1, 2), Vector2(3, 4), Vector2(5, 6)),
		Vector4(2.5, 5, 10, 20), Vector4i(5, 10, 20, 40)
	)


func test_math_types_1_out_2():
	res.math_args_1(
		res.vector2_ret(), res.vector2i_ret(),
		res.rect2_ret(), res.rect2i_ret(),
		res.vector3_ret(), res.vector3i_ret(),
		res.transform2d_ret(),
		res.vector4_ret(), res.vector4i_ret()
	)


func test_math_types_2_out():
	res.math_args_2(
		res.plane_ret(), res.quaternion_ret(), res.aabb_ret(), res.basis_ret(),
		res.transform3d_ret(), res.projection_ret(), res.color_ret()
	)


func test_callable():
	return true


func test_misc_types_out():
	# print(res.get_rid(), res.object_ret(), test_callable, test_signal)

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
