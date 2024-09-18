extends SceneTree

func _init():
	var console = Python.import_module("console")

	console.getattr("interact").call([Engine.get_version_info()], {})

	quit()
