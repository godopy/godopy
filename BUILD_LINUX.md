## Installing dependencies
```
sudo apt-get install python3-venv python3-dev
sudo apt-get build-dep python3
```


## Building and setting up PyGodot development environment
```
$ cd pygodot
$ ./internal_python_build.py
$ deps/python/build/bin/python3 -m venv buildenv
$ source buildenv/bin/activate
(buildenv) $ python -m pip install --upgrade pip
(buildenv) $ pip install deps/cython
(buildenv) $ pip install -r internal-requirements.txt
(buildenv) $ deactivate
$ cd ..
$ python3 -m venv toolbox
$ source toolbox/bin/activate
(toolbox) $ python -m pip install --upgrade pip
(toolbox) $ pip install -r pygodot/requirements.txt
(toolbox) $ export GODOT_BUILD=<path to Godot source folder>
(toolbox) $ cd pygodot
(toolbox) $ ./bootstrap.py
(toolbox) $ pip install -e .
(toolbox) $ scons  # scons -j4 only_cython=yes && scons -j4
(toolbox) $ pip install -e .
(toolbox) $ cd ..
```
> Replace `<path to Godot source folder>` with an actual path. Godot source should be compiled.
> When you finish working with a virtual environment, run `deactivate` command
> Cython installation before other packages ensures that their build process will use the same version of Cython
> If you want a faster parallel initial build, build with "only_cython=yes" first, otherwise the required headers will be missing
