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
> cd godopy
> py build_python.py
> deps\python\PCbuild\amd64\python.exe -m venv venv
> venv\Scripts\activate
(venv) > cp deps\python\PC\pyconfig.h venv\Include\
(venv) > py -m pip install -U pip Cython numpy ipython
(venv) > # Use pip to install any Python dependencies you want
(venv) > deactivate
```


## Building GodoPy
```
> py -m venv setup
> setup\Scripts\activate
(setup) > py -m pip install -r dev-requirements.txt
(setup) > py bootstrap.py
(setup) > venv\Scripts\activate
(venv) > py -m pip install .\_lib
(venv) > setup\Scripts\activate
(setup) > scons
(setup) > deactivate
```
> If you want to run an initial build with a -j option, build with "only_cython=yes" first, otherwise the required headers will be missing
> Python wheels inside `dist/` can be created with `py -m pep517.build .` command

## Setting up GodoPy development environment
```
> cd ..  # return to the project's root
> py -m venv tools
> tools\Scripts\activate
(tools) $ cd godopy
(tools) > py -m pip install -r dev-requirements.txt
(tools) > py setup.py develop
```
> When you finish working with a virtual environment, run the `deactivate` command
