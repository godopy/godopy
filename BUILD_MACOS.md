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
$ 3rdparty/python/build/bin/python3 -m venv env
$ source env/bin/activate
(env) $ pip install -U pip Cython numpy
(env) $ # Install any Python dependencies you'd like to use inside the engine
```


## Building GodoPy
```
$ python -m venv gdpy-setup
$ source gdpy-setup/bin/activate
(gdpy-setup) $ pip install -U pip -r setup-requirements.txt
(gdpy-setup) $ ./bootstrap.py
(gdpy-setup) $ ./clean.sh env
(gdpy-setup) $ scons --jobs=$(sysctl -n hw.logicalcpu) only_cython=yes
(gdpy-setup) $ scons --jobs=$(sysctl -n hw.logicalcpu)
```


## Setting up GodoPy development environment
```
$ python -m venv ../meta
$ source ../meta/bin/activate
(meta) $ python setup.py develop
```
