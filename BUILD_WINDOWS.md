## Installing dependencies

```
choco install python
choco install mingw  # For C preprocessor (cpp.exe)
set-executionpolicy RemoteSigned  # To enable virtualenv activation
```

## Building and setting up PyGodot development environment
```
> cd pygodot
> python internal_python_build.py
> # python internal_python_build.py target=debug
> .\deps\python\PCbuild\amd64\python.exe -m venv .\buildenv
> .\buildenv\Scripts\activate
(buildenv) > cp .\deps\python\PC\pyconfig.h .\buildenv\Include\
(buildenv) > python -m pip install deps\cython
(buildenv) > python -m pip install -r internal-packages\requirements.txt
(buildenv) > deactivate
> cd ..
> python -m venv toolbox
> .\toolbox\Scripts\activate
(toolbox) > python -m pip install -r pygodot\requirements.txt
(toolbox) > $env:GODOT_BUILD = 'C:\path\to\godot'
(toolbox) > cd pygodot
(toolbox) > python bootstrap.py
(toolbox) > python -m pip install -e .
(toolbox) > scons  # scons -j4 only_cython=yes && scons -j4
(toolbox) > python -m pip install -e .
(toolbox) > cd ..
```
> Replace `C:\path\to\godot` with an actual path.
> When you finish working with a virtual environment, run `deactivate` command
> Cython installation before other packages ensures that their build process will use the same version of Cython
