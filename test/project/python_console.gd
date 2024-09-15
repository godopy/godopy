extends SceneTree

func _init():
	var console = Python.import_module("console")

	console.getattr("interact").call([], {
		"godot_banner":
			"Enhanced by Godot Engine version %s GDExtension API (%s [%s])" % [
				OS.get_version(),
				OS.get_name(),
				Engine.get_architecture_name()
			]
	})

	quit()
