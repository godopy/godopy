extends SceneTree

func _init():
	Python.run_simple_string("print(\"Test 'run_simple_string(str)'\")\n")
	quit()
