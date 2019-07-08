## Compile GodotPython.pyx

```sh
cd src/core
cython -3 --cplus -o GodotPython.cpp GodotPython.pyx
mv GodotPython.h ../../include/core
```

## Generate cusom godot_headers

`../godot` is a path to the Godot build

```sh
cp -R ../godot/modules/gdnative/include ./godot_headers
../godot/bin/godot.osx.opt.tools.64 --gdnative-generate-json-api godot_headers/api.json
```

## Generate gdnative_api_struct__gen.pxd

`autopxd2` must be installed in the active venv. Manually change `autopxd2` to use `clang++ -E` instead of
`cpp`.

```sh
cd godot_headers
autopxd -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include gdnative_api_struct.gen.h ../godot/gdnative_api_struct__gen.pxd
```

### Edit `gdnative_api_struct__gen.pxd`

Add the following c-imports:

```cython
from libc.stddef cimport wchar_t
from libcpp cimport bool
```

Mark the definitions as `nogil`:

```cython
cdef extern from "gdnative_api_struct.gen.h" nogil:
```

Find and replace `uint8_t _dont_touch_that[]` with `pass`
