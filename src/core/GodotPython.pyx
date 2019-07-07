from godot.core cimport String, Godot

cdef void print(str s):
    py_message = s.encode('utf-8')
    cdef String message = <const char *> py_message
    Godot.print(message)

cdef public void gdpython_print_banner():
    import sys

    print("GodotPython 0.0.1a.dev~");
    print(f"Python {sys.version}")
