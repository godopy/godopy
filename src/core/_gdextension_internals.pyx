from libc.stdint cimport int32_t, uint64_t
from cpython cimport PyObject, PyUnicode_FromWideChar
from binding cimport (
    gdextension_interface_print_error as print_error
)
from godot_cpp cimport OS, String, CharWideString


# Don't rely on any automatic type conversions.
# Python error printing should always work even when other modules don't.
# `godot_cpp.UtilityFunctions.*` would perform implicit Python->Variant conversions,
# therefore we can't use them here.
cdef extern from *:
    """
    void __debug_print(PyObject *p_msg) {
        const wchar_t *wstr = PyUnicode_AsWideCharString(p_msg, NULL);
        godot::String msg;
        godot::internal::gdextension_interface_string_new_with_wide_chars(&msg, wstr);
        godot::UtilityFunctions::print(msg);
    }

    void __print_rich(PyObject *p_msg) {
        const wchar_t *wstr = PyUnicode_AsWideCharString(p_msg, NULL);
        godot::String msg;
        godot::internal::gdextension_interface_string_new_with_wide_chars(&msg, wstr);
        godot::UtilityFunctions::print_rich(msg);
    }

    void __push_error(PyObject *p_msg) {
        const wchar_t *wstr = PyUnicode_AsWideCharString(p_msg, NULL);
        godot::String msg;
        godot::internal::gdextension_interface_string_new_with_wide_chars(&msg, wstr);
        godot::UtilityFunctions::push_error(msg);
    }
    """
    cdef void __debug_print(str) noexcept
    cdef void __print_rich(str) noexcept
    cdef void __push_error(str) noexcept


import io, os, sys, traceback
from types import ModuleType


if sys.platform == 'linux':
    # Help numpy find libscipy_openblas*.so library
    import glob
    paths = glob.glob(f"{sys.exec_prefix}/libscipy_openblas*")
    if paths:
        import ctypes
        cdll = ctypes.CDLL(paths[0])


# Allow numpy dynamic libraries to exist separately from pure Python code.
# This requires some acrobatics to get right.
try:
    _numpy = ModuleType('numpy')
    _core = ModuleType('numpy._core')

    # Set placeholder __path__s to include all required files
    _numpy.__path__ = [os.path.join(sys.path[2], 'numpy')]
    _core.__path__ = [os.path.join(sys.path[2], 'numpy', '_core'), sys.exec_prefix]

    sys.modules['numpy'] = _numpy
    sys.modules['numpy._core'] = _core

    import _multiarray_umath
    import _simd

    sys.modules['numpy._core._multiarray_umath'] = _multiarray_umath
    sys.modules['numpy._core._simd'] = _simd
    del sys.modules['numpy._core']

    _linalg = ModuleType('numpy.linalg')
    _linalg.__path__ = [os.path.join(sys.path[2], 'numpy', 'linalg'), sys.exec_prefix]
    sys.modules['numpy.linalg'] = _linalg

    import _umath_linalg
    import lapack_lite
    sys.modules['numpy.linalg._umath_linalg'] = _umath_linalg
    sys.modules['numpy.linalg.lapack_lite'] = lapack_lite
    del sys.modules['numpy.linalg']

    _fft = ModuleType('numpy.fft')
    _fft.__path__ = [os.path.join(sys.path[2], 'numpy', 'fft'), sys.exec_prefix]
    sys.modules['numpy.fft'] = _fft

    import _pocketfft_umath
    sys.modules['numpy.fft._pocketfft_umath'] = _pocketfft_umath
    del sys.modules['numpy.fft']
    del sys.modules['numpy']

    import numpy
    import numpy._core
    import numpy.linalg
    import numpy.fft

    numpy._core._multiarray_umath = _multiarray_umath

    import numpy.dtypes as dtypes
    numpy.dtypes = dtypes

    # Round 2: numpy.random.* dynamic modules fix
    random = ModuleType('numpy.random')

    # Set the __path__ of the placeholder module to include all numpy.random modules, including .pyd files
    random.__path__ = [os.path.join(numpy.__path__[0], 'random'), sys.exec_prefix]
    sys.modules['numpy.random'] = random

    # These imports must be relative, otherwise relative imports inside of them won't work
    from numpy.random import mtrand
    from numpy.random import _generator

    sys.modules['numpy.random.mtrand'] = mtrand
    sys.modules['numpy.random._generator'] = _generator

    import bit_generator
    import _bounded_integers
    import _common
    import _mt19937
    import _pcg64
    import _philox
    import _sfc64

    sys.modules['numpy.random.bit_generator'] = bit_generator
    sys.modules['numpy.random._bounded_integers'] = _bounded_integers
    sys.modules['numpy.random._common'] = _common
    sys.modules['numpy.random._mt19937'] = _mt19937
    sys.modules['numpy.random._pcg64'] = _pcg64
    sys.modules['numpy.random._philox'] = _philox
    sys.modules['numpy.random._sfc64'] = _sfc64

    # Remove the placeholder and reimport
    del sys.modules['numpy.random']
    import numpy.random

    # for mod_name, mod in sys.modules.items():
    #     if 'numpy' in mod_name:
    #         __debug_print("\t%r: %r" % (mod_name, mod.__file__))

except Exception as exc:
    print_traceback_and_die(exc)


cdef int print_traceback(object exc) except -1:
    cdef object f = io.StringIO()

    traceback.print_exception(exc, file=f)
    cdef object exc_text = f.getvalue()

    cdef bytes descr, path, func
    cdef int32_t line

    try:
        descr = str(exc).encode('utf-8')

        exc_lines = exc_text.splitlines()
        if len(exc_lines) > 1:
            info_line_str = exc_lines[-2]

            if info_line_str.lstrip().startswith('^'):
                info_line_str = exc_lines[-4]
            elif not (info_line_str.lstrip().startswith('File') and 'line' in info_line_str):
                info_line_str = exc_lines[-3]

            info_line = [s.strip().strip(',') for s in info_line_str.split()]

            path = info_line[1].encode('utf-8')
            func = info_line[-1].encode('utf-8')
            line = int(info_line[3])
        else:
            info_line_str = exc_lines[0]
            info_line = [s.strip().strip(':') for s in info_line_str.split()]
            path_line = info_line[1].rsplit(':', 1)
            path = path_line[0].encode('utf-8')
            func = b''
            line = int(path_line[1])

        __print_rich("[color=purple]%s[/color]" % exc_text)
        print_error(descr, path, func, line, False)

    except Exception as exc2:
        __print_rich("[color=red]Exception parser raised an exception: %s[/color]" % exc2)

        __print_rich("[color=purple]%s[/color]" % exc_text)
        __push_error(str(exc))

    return 0


cdef public void print_traceback_and_die(object exc) noexcept:
    print_traceback(exc)

    # Kill the process immediately to prevent a flood of unrelated errors
    cdef uint64_t pid = OS.get_singleton().get_process_id()
    OS.get_singleton().kill(pid)


default_gdextension_config = {
    'registered_modules': ('godot', 'godot_scripting', 'godopy'),
}
