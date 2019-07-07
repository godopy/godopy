## Compile GodotPython.pyx

```
cd path/to/src/core
cython -3 --cplus -o GodotPython.cpp GodotPython.pyx
mv GodotPython.h ../../include/core
```
