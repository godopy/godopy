## Installing dependencies

```
choco install python
choco install mingw  # For C preprocessor (cpp.exe)
set-executionpolicy RemoteSigned  # To enable virtualenv activation
```

## Setting environment variables
```
> $env:GODOT_BUILD = 'C:\path\to\godot'
```
> Replace `C:\path\to\godot` with an actual path. Godot source should be compiled.


## Building internal Python interpreter and libraries
```
> cd GodoPy
> py build_python.py
> deps\python\PCbuild\amd64\python.exe -m venv env
> env\Scripts\activate
(env) > cp deps\python\PC\pyconfig.h env\Include\
(env) > py -m pip install -U pip Cython numpy
(env) > # Install any Python dependencies you'd like to use inside the engine
```


## Building GodoPy
```
> py -m venv gdpy-setup
> gdpy-setup\Scripts\activate
(gdpy-setup) > py -m pip install -U pip -r setup-requirements.txt
(gdpy-setup) > py bootstrap.py
(gdpy-setup) > env\Scripts\activate
(env) > py -m pip install .\_lib
(env) > gdpy-setup\Scripts\activate
(gdpy-setup) > scons only_cython=yes
(gdpy-setup) > scons
```


## Setting up GodoPy development environment
```
> py -m venv ..\meta
> ..\meta\Scripts\activate
(meta) > py -m pip install -r dev-requirements.txt
(meta) > py setup.py develop
```
