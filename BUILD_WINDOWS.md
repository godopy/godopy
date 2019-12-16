## Installing dependencies

```
choco install python
choco install mingw  # For C preprocessor (cpp.exe)
set-executionpolicy RemoteSigned  # To enable virtualenv activation
```

## Setting environment variables
```
> $env:GODOT_BUILD = 'C:\path\to\godot'<path to Godot source folder>
```
> Replace `C:\path\to\godot` with an actual path. Godot source should be compiled.


## Building internal Python interpreter and libraries
```
> cd pygodot
> python internal_python_build.py
> .\deps\python\PCbuild\amd64\python.exe -m venv .\buildenv
> .\buildenv\Scripts\activate
(buildenv) > cp .\deps\python\PC\pyconfig.h .\buildenv\Include\
(buildenv) > pip install -r internal-requirements.txt
(buildenv) > deactivate
> cd ..
```


## Setting up PyGodot development environment
```
> python -m venv toolbox
> .\toolbox\Scripts\activate
(toolbox) > pip install -r pygodot\bootstrap-requirements.txt
(toolbox) > cd pygodot
(toolbox) > python bootstrap.py
(toolbox) > deactivate
> .\buildenv\Scripts\activate
(buildenv) > pip install .\internal-packages
(buildenv) > deactivate
> ..\toolbox\Scripts\activate
(toolbox) > scons
(toolbox) > pip install -e .
(toolbox) > cd ..
```

> When you finish working with a virtual environment, run `deactivate` command
> Cython installation before other packages ensures that their build process will use the same version of Cython
