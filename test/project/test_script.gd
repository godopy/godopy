extends SceneTree

func _init():
	Python.run_simple_string(
		"import godot\n" +
		"godot.redirect_python_stdout()\n" +
		"import sys\n" +
		"for p in sys.path:\n" +
		"    print(p)\n" +
		# "import numpy as np\n" +
		# "a1D = np.array([1, 2, 3, 4])\n" +
		# "print(a1D)\n" +
		"raise Exception('Test exc')"
	)

	Python.run_simple_string("print('Hello, world!')")

	quit()