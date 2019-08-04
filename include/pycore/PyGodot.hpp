#ifndef PYGODOT_HPP
#define PYGODOT_HPP

#include "PythonGlobal.hpp"

#include "Godot.hpp"

namespace pygodot {

typedef godot_variant (*__godot_wrapper_method)(godot_object *, void *, void *, int, godot_variant **);

inline const char *__get_python_class_name(PyTypeObject *cls) {
  ERR_FAIL_PYTHON_NULL_V(cls, NULL);
  PyObject *u_class_name = PyObject_GetAttrString((PyObject *)cls, "__name__"); ERR_FAIL_PYTHON_NULL_V(u_class_name, NULL);
  PyObject *b_class_name = PyUnicode_AsUTF8String(u_class_name); ERR_FAIL_PYTHON_NULL_V(b_class_name, NULL);
  const char *class_name = (const char *)PyBytes_AsString(b_class_name); ERR_FAIL_PYTHON_NULL_V(class_name, NULL);

  return class_name;
}

// Adaptation of C++ templates from Godot.hpp to Cython functions

// Self is `UserDefinedClass *`, R is a return type (PyObject * by default)
template <class Self, class R, class... As>
struct _WrappedMethod {
  R(*f)(Self, As...);

  template <int... I>
  void apply(godot::Variant *ret, Self obj, godot::Variant **args, godot::__Sequence<I...>) {
    // FIXME: NULL and PyErr_Occurred checks?
    *ret = (*f)(obj, godot::_ArgCast<As>::_arg_cast(*args[I])...);
  }
};

template <class Self, class... As>
struct _WrappedMethod<Self, PyObject *, As...> {
  PyObject *(*f)(Self, As...);

  template <int... I>
  void apply(godot::Variant *ret, Self obj, godot::Variant **args, godot::__Sequence<I...>) {
    PyObject *__result = (*f)(obj, godot::_ArgCast<As>::_arg_cast(*args[I])...);
    ERR_FAIL_PYTHON_NULL(__result);
    *ret = __result;
  }
};

template <class Self, class... As>
struct _WrappedMethod<Self, void, As...> {
  void (*f)(Self, As...);

  template <int... I>
  void apply(godot::Variant *ret, Self obj, godot::Variant **args, godot::__Sequence<I...>) {
    (*f)(obj, godot::_ArgCast<As>::_arg_cast(*args[I])...);
  }
};

template <class Self, class R, class... As>
godot_variant __wrapped_method(godot_object *, void *method_data, void *user_data, int num_args, godot_variant **args) {
  godot_variant v;
  godot::api->godot_variant_new_nil(&v);

  Self obj = (Self)user_data;
  _WrappedMethod<Self, R, As...> *method = (_WrappedMethod<Self, R, As...> *)method_data;

  godot::Variant *var = (godot::Variant *)&v;
  godot::Variant **arg = (godot::Variant **)args;

  method->apply(var, obj, arg, typename godot::__construct_sequence<sizeof...(As)>::type{});

  return v;
}

template <class Self, class R, class... As>
void *___make_wrapper_function(R (*f)(Self, As...)) {
  using MethodType = _WrappedMethod<Self, R, As...>;
  MethodType *p = (MethodType *)godot::api->godot_alloc(sizeof(MethodType));
  p->f = f;
  return (void *)p;
}

template <class Self, class R, class... As>
__godot_wrapper_method ___get_wrapper_function(R (*f)(Self, As...)) {
  return (__godot_wrapper_method)&__wrapped_method<Self, R, As...>;
}

template <class M>
PyObject *register_method(PyTypeObject *cls, const char *name, M method_ptr, godot_method_rpc_mode rpc_type=GODOT_METHOD_RPC_MODE_DISABLED) {
  godot_instance_method method = {};
  method.method_data = ___make_wrapper_function(method_ptr);
  method.free_func = godot::api->godot_free;
  method.method = (__godot_wrapper_method)___get_wrapper_function(method_ptr);

  godot_method_attributes attr = {};
  attr.rpc_type = rpc_type;

  godot::nativescript_api->godot_nativescript_register_method(godot::_RegisterState::nativescript_handle, __get_python_class_name(cls), name, attr, method);

  Py_INCREF(Py_None);
  return Py_None;
}

} // namespace pygodot

#endif // PYGODOT_HPP
