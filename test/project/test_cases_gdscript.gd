extends Node

var res = TestResource.new()

func get_resource():
	return res

func test_atomic_types():
	res.atomic_args(res.bool_ret(), res.int_ret(), res.float_ret(), res.string_ret())

func test_math_types_1():
	res.math_args_1(Vector2(2.5, 5), Vector2i(5, 10), Rect2(0, 0, 100, 200), Rect2i(0, 0, 100, 200))
