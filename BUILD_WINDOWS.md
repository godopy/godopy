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
> 3rdparty\python\PCbuild\amd64\python.exe -m venv .env
> .env\Scripts\activate
(.env) > cp deps\python\PC\pyconfig.h env\Include\
(.env) > py -m pip install -U pip Cython numpy
(.env) > # Install any Python dependencies you'd like to use inside the engine
```


## Building GodoPy
```
> py -m venv .setup
> .setup\Scripts\activate
(.setup) > py -m pip install -U pip -r setup-requirements.txt
(.setup) > py bootstrap.py
(.setup) > .env\Scripts\activate
(.env) > py -m pip install .\_lib
(.env) > .setup\Scripts\activate
(.setup) > scons only_cython=yes
(.setup) > scons
```


## Setting up GodoPy development environment

Create "meta" virtual environment:
```
> py -m venv ..\meta
> ..\meta\Scripts\activate
```
> "meta" is an example, any other name would work ("venv", "my-game" etc)
> you may activate an existing virtual environment, GodoPy will be installed there

Build and link GodoPy inside created virtualenv:
```
(meta) > py setup.py develop
```
