extends SceneTree

func _init():
	var console = PythonRuntime.import_module("godopy.contrib.console.terminal")

	console.getattr("interact").call([], {})

	quit()
