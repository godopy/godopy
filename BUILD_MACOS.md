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
(venv) $ pip install -r _lib/requirements.txt
(venv) $ # Use pip to install any Python dependencies you want
(venv) $ deactivate
```

## Building GodoPy
```
$ python3 -m venv setup
$ source setup/bin/activate
(setup) $ pip install -r requirements.txt
(setup) $ ./bootstrap.py && ./clean.sh
(setup) $ scons  # scons -j4 only_cython=yes && scons -j4
(setup) $ deactivate
```
> Sometimes it is required to deactivate and reactivate the virtual environment before running scons
> If you want to run an initial build with a -j option, build with "only_cython=yes" first, otherwise the required headers will be missing


## Setting up GodoPy development environment
```
$ cd .. # return to the project's root
$ python3 -m venv toolbox
$ source toolbox/bin/activate
(toolbox) $ pip install -e ./godopy  # path to GodoPy build
```
> When you finish working with a virtual environment, run `deactivate` command