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
> deps\python\PCbuild\amd64\python.exe -m venv godopy-venv
> godopy-venv\Scripts\activate
(godopy-venv) > cp deps\python\PC\pyconfig.h venv\Include\
(godopy-venv) > py -m pip install -U pip Cython numpy
(godopy-venv) > # Use pip to install any Python dependencies you want
(godopy-venv) > deactivate
```


## Building GodoPy
```
> py -m venv godopy-build-venv
> setup\Scripts\activate
(godopy-build-venv) > py -m pip install -r dev-requirements.txt
(godopy-build-venv) > py bootstrap.py
(godopy-build-venv) > godopy-build-venv\Scripts\activate
(godopy-build-venv) > py -m pip install .\_lib
(godopy-build-venv) > godopy-build-venv\Scripts\activate
(godopy-build-venv) > scons
```
> Python wheels inside `dist/` can be created with `py -m pep517.build .` command

## Setting up GodoPy development environment
```
> py -m venv ..\meta
> ..\meta\Scripts\activate
(meta) > py -m pip install -r dev-requirements.txt
(meta) > py setup.py develop
```
