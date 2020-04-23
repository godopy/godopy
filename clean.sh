rm _lib/godot/*.cpp
rm _lib/godot/core/*.cpp
rm _lib/godot/bindings/*.cpp
source godopy-venv/bin/activate
pip install ./_lib
