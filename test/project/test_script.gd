extends SceneTree

func _init():
	Python.run_simple_string("import godot; godot.print('[color=green]Hello, world![/color]')\n")
	quit()
