## Generate cusom godot_headers

`../godot` is the path to the Godot build

```sh
cp -R ../godot/modules/gdnative/include ./godot_headers
../godot/bin/godot.osx.opt.tools.64 --gdnative-generate-json-api godot_headers/api.json
touch godot_headers/__init__.py
```

## Generate gdnative_api_struct__gen.pxd

`autopxd2` must be installed in the active venv. Manually change `autopxd2` to use `clang++ -E` instead of
`cpp`.

```sh
cd godot_headers
autopxd -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include gdnative_api_struct.gen.h ../godot_headers/gdnative_api_struct__gen.pxd
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


## Initialize Cython bindings

Inside venv:

```sh
pip install -e .
```
