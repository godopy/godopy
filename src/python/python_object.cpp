
#include "python_object.h"
#include "python_runtime.h"
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

PythonObject::PythonObject() {
    instance = NULL;
}

PythonObject::~PythonObject() {
    Py_XDECREF(instance);
}

Variant PythonObject::call(const Array &p_args, const Dictionary &p_kwargs) {
    Variant ret;
    PyGILState_STATE gil_state = PyGILState_Ensure();

    ERR_FAIL_NULL_V(instance, ret);

    if (!PyCallable_Check(instance)) {
        PyGILState_Release(gil_state);
        UtilityFunctions::push_error("PythonObject '" + __repr__ + "' is not callable.");
        return ret;
    }

    PyObject *args = PyTuple_New(p_args.size());
    ERR_FAIL_NULL_V(args, ret);

    for (size_t i = 0; i < p_args.size(); i++) {
        Variant arg = p_args[i];
        PyTuple_SetItem(args, i, arg.pythonize());
    }

    PyObject *kwargs = nullptr;
    const Array keys = p_kwargs.keys();
    if (keys.size() > 0) {
        kwargs = PyDict_New();
        ERR_FAIL_NULL_V(kwargs, ret);

        for (size_t i = 0; i < keys.size(); i++) {
            Variant k = keys[i];
            PyObject *key = k;
            ERR_FAIL_NULL_V(key, nullptr);
            PyObject *val = p_kwargs[k];
            ERR_FAIL_NULL_V(val, nullptr);
            PyDict_SetItem(kwargs, key, val);
        }
    }

    PyObject *result = PyObject_Call(instance, args, kwargs);

    if (result == nullptr) {
		if (PyErr_Occurred()) {
            PyErr_Print();
            PyErr_Clear();
        }
	} else {
        result = Variant(result);
    }
    Py_XDECREF(result);
    PyGILState_Release(gil_state);

    return ret;
}


Variant PythonObject::call_one_arg(const Variant &p_arg) {
    Variant ret;
    PyGILState_STATE gil_state = PyGILState_Ensure();

    ERR_FAIL_NULL_V(instance, ret);

    if (!PyCallable_Check(instance)) {
        PyGILState_Release(gil_state);
        UtilityFunctions::push_error("PythonObject '" + __repr__ + "' is not callable.");
        return ret;
    }

    PyObject *arg = p_arg.pythonize();
    PyObject *result = PyObject_CallOneArg(instance, arg);

    if (result == nullptr) {
		if (PyErr_Occurred()) {
            PyErr_Print();
            PyErr_Clear();
        }
	} else {
        result = Variant(result);
    }
    Py_XDECREF(result);
    PyGILState_Release(gil_state);

    return ret;
}

Ref<PythonObject> PythonObject::getattr(const String &p_attr_name) {
    Ref<PythonObject> object = memnew(PythonObject);
    object->__name__ = p_attr_name;

    ERR_FAIL_NULL_V(instance, object);

    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyObject *attr = PyObject_GetAttrString(instance, p_attr_name.utf8());
    if (PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
    ERR_FAIL_NULL_V(attr, object);
    Py_INCREF(attr);
    object->instance = attr;
    PyObject *repr = PyObject_Repr(attr);
    ERR_FAIL_NULL_V(attr, object);
    object->__repr__ = String(repr);
    PyGILState_Release(gil_state);

    return object;
}

bool PythonObject::is_callable() {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    bool is_callable = PyCallable_Check(instance);
    PyGILState_Release(gil_state);
    return is_callable;
}

void PythonObject::_bind_methods() {
    ClassDB::bind_method(D_METHOD("getattr", "string"), &PythonObject::getattr);
    ClassDB::bind_method(D_METHOD("is_callable"), &PythonObject::is_callable);
    ClassDB::bind_method(D_METHOD("call", "array", "dictionary"), &PythonObject::call);

    // FIXME: Can not compile call_varargs bind
    // MethodInfo mi;
	// mi.name = "call_varargs";
    // std::vector<Variant> v;
	// ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "call_varargs", &PythonObject::call_varargs, mi, v, true);
}
