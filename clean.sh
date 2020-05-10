rm _lib/godot/*.cpp
rm _lib/godot/core/*.cpp
rm _lib/godot/bindings/*.cpp
source $1/bin/activate
pip install ./_lib
