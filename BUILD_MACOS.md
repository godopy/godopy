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
$ cd godopy
$ ./build_python.py
$ deps/python/build/bin/python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r batteries/requirements.txt
(venv) $ # Use pip to install any Python dependencies you want
(venv) $ deactivate
$ cd ..
```

## Setting up GodoPy development environment
```
$ python3 -m venv toolbox
$ source toolbox/bin/activate
(toolbox) $ pip install -r godopy/requirements.txt
(toolbox) $ cd godopy
(toolbox) $ ./bootstrap.py && ./clean.sh
(toolbox) $ scons  # scons -j4 only_cython=yes && scons -j4
(toolbox) $ pip install -e .
(toolbox) $ cd ..
```
> When you finish working with a virtual environment, run `deactivate` command
> Cython installation before other packages ensures that their build process will use the same version of Cython
> If you want a faster parallel initial build, build with "only_cython=yes" first, otherwise the required headers will be missing
