extends SceneTree

func _init():
	var console = Python.import_module("godopy.contrib.console.terminal")

	console.getattr("interact").call([Engine.get_version_info()], {})

	quit()
