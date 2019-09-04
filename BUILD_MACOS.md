## Installing dependencies
```
brew install python3 openssl sqlite3
```

## Setting environment variables
```
export GODOT_BUILD=<path to Godot source folder>
```
> Replace `<path to Godot source folder>` with an actual path. Godot source should be compiled.


## Building internal Python interpreter and libraries
```
$ cd pygodot
$ ./internal_python_build.py
$ deps/python/build/bin/python3 -m venv buildenv
$ source buildenv/bin/activate
(buildenv) $ pip install deps/cython
(buildenv) $ pip install -r internal-requirements.txt
(buildenv) $ deactivate
$ cd ..
```

## Setting up PyGodot development environment
```
$ python3 -m venv toolbox
$ source toolbox/bin/activate
(toolbox) $ pip install -r pygodot/bootstrap-requirements.txt
(toolbox) $ deactivate
$ source toolbox/bin/activate
(toolbox) $ cd pygodot
(toolbox) $ ./clean.sh
(toolbox) $ ./bootstrap.py
(toolbox) $ scons  # scons -j4 only_cython=yes && scons -j4
(toolbox) $ pip install -e .
(toolbox) $ cd ..
```
> When you finish working with a virtual environment, run `deactivate` command
> Cython installation before other packages ensures that their build process will use the same version of Cython
> If you want a faster parallel initial build, build with "only_cython=yes" first, otherwise the required headers will be missing
