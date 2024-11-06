from godot.classdb import Engine
from godopy.contrib.console import terminal


if __name__ == '__main__':
	terminal.interact(Engine.get_version_info())
