#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/variant.hpp>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

extern int print_traceback(PyObject *);

using namespace godot;

class PythonObject : public RefCounted {
	GDCLASS(PythonObject, RefCounted);

    friend class PythonRuntime;
    friend class Variant;
private:
    PyObject *instance;
    String __name__;
    String __repr__;

    _ALWAYS_INLINE_ void set_name(const String &p_name) { __name__ = p_name; }
    _ALWAYS_INLINE_ void set_repr(const String &p_repr) { __repr__ = p_repr; }
    _ALWAYS_INLINE_ void set_instance(PyObject *p_instance) { instance = p_instance; }

protected:
	static void _bind_methods();

public:
    PythonObject();
    ~PythonObject();

    template <typename... Args>
	Variant call_varargs(const Args &...p_args) {
		std::array<Variant, sizeof...(Args)> variant_args{ Variant(p_args)... };
        TypedArray<Variant *> call_args;
        call_args.resize(sizeof...(Args));
        for (size_t i = 0; i < variant_args.size(); i++) {
            call_args[i] = &variant_args[i];
        }
        return call(call_args);
	}

    Variant call(const Array &p_args = Array(), const Dictionary &p_kwargs = Dictionary());
    Variant call_one_arg(const Variant &p_arg);

    Ref<PythonObject> getattr(const String &);
    bool is_callable();
};
