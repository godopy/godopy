## Installing dependencies
```
brew install python3 openssl sqlite3 zlib readline
```

## Setting environment variables
```
export GODOT_BUILD=<path to Godot source folder>
```
> Replace `<path to Godot source folder>` with an actual path. Godot source should be compiled.


## Building internal Python interpreter and libraries
```
$ cd GodoPy
$ ./build_python.py
$ deps/python/build/bin/python3 -m venv godopy-venv
$ source godopy-venv/bin/activate
(godopy-venv) $ pip install -U pip Cython numpy
(godopy-venv) $ # Use pip to install any Python dependencies you want
```

## Building GodoPy
```
$ python -m venv godopy-build-venv
$ source godopy-build-venv/bin/activate
(godopy-build-venv) $ pip install -U pip -r dev-requirements.txt
(godopy-build-venv) $ ./bootstrap.py
(godopy-build-venv) $ ./clean.sh
(godopy-build-venv) $ scons --jobs=$(sysctl -n hw.logicalcpu) only_cython=yes
(godopy-build-venv) $ scons --jobs=$(sysctl -n hw.logicalcpu) 
```
> Python wheels inside `dist/` can be created with `python -m pep517.build .` command


## Setting up GodoPy development environment
```
$ python -m venv ../meta
$ source ../meta/bin/activate
(meta) $ python setup.py develop
```
