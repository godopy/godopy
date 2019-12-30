#include "PyGodot.hpp"

#include <batteries/godot/nativescript.hpp>

namespace godopy {

PyObject *register_method(PyTypeObject *cls, const char *name, PyObject *method_ptr, godot_method_rpc_mode rpc_type) {
  PyObject *ret = _register_python_method(cls, name, method_ptr, rpc_type);
  ERR_FAIL_PYTHON_NULL_V(ret, NULL);

  return ret;
}

PyObject *register_property(PyTypeObject *cls, const char *name, PyObject *default_value, godot_method_rpc_mode rpc_mode,
                            godot_property_usage_flags usage, godot_property_hint hint, godot::String hint_string) {
  PyObject *ret = _register_python_property(cls, name, default_value, rpc_mode, usage, hint, hint_string);
  ERR_FAIL_PYTHON_NULL_V(ret, NULL);

  return ret;
}

PyObject *register_signal(PyTypeObject *cls, PyObject *name, PyObject *args) {
  PyObject *ret = _register_python_signal(cls, name, args);
  ERR_FAIL_PYTHON_NULL_V(ret, NULL);

  return ret;
}

}