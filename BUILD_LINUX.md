## Installing dependencies
```
sudo apt-get install python3-venv python3-dev
```
If not already done, enable the Ubuntu source 'deb-src' repository in 'Software and Updates' before installing these python deps: 
```
sudo apt-get build-dep python3.8
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
$ 3rdparty/python/build/bin/python3 -m venv .env
$ source env/bin/activate
(.env) $ pip install -U pip Cython numpy
(.env) $ # Install any Python dependencies you'd like to use inside the engine
```


## Building GodoPy
```
$ python -m venv .setup
$ source gdpy-setup/bin/activate
(.setup) $ pip install -U pip -r dev-requirements.txt
(.setup) $ ./bootstrap.py
(.setup) $ ./clean.sh .env
(.setup) $ scons only_cython=yes
(.setup) $ scons
```


## Setting up GodoPy development environment
Create "meta" virtual environment:
```
$ python3 -m venv ../meta
$ source ../meta/bin/activate
```
> "meta" is an example, any other name would work ("venv", "my-game" etc)
> you may activate an existing virtual environment, GodoPy will be installed there

Build and link GodoPy inside created virtualenv:
```
(meta) $ python setup.py develop
```
