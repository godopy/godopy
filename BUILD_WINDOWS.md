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
> python build_python.py
> deps\python\PCbuild\amd64\python.exe -m venv .\venv
> venv\Scripts\activate
(venv) > cp .\deps\python\PC\pyconfig.h .\venv\Include\
(venv) > pip install -r _lib\requirements.txt
(venv) > # Use pip to install any Python dependencies you want
(venv) > deactivate
```


## Building GodoPy
```
> python -m venv setup
> setup\Scripts\activate
(setup) > pip install -r requirements.txt
(setup) > python bootstrap.py
(setup) > venv\Scripts\activate
(venv) > pip install .\_lib
(venv) > setup\Scripts\activate
(setup) > scons
(setup) > deactivate
```
> If you want to run an initial build with a -j option, build with "only_cython=yes" first, otherwise the required headers will be missing


## Setting up GodoPy development environment
```
> python -m venv toolbox
> toolbox\Scripts\activate
(toolbox) $ cd godopy
(toolbox) > pip install -r requirements.txt
(toolbox) > python setup.py develop
```
> When you finish working with a virtual environment, run `deactivate` command
