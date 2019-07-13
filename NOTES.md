## Generate godot_headers

`../godot` is the path to the Godot build

```sh
cp -R ../godot/modules/gdnative/include ./godot_headers
../godot/bin/godot.osx.opt.tools.64 --gdnative-generate-json-api godot_headers/api.json
pygodot genapi
```

## Initialize Cython bindings

Inside venv:

```sh
pip install -e .
```
