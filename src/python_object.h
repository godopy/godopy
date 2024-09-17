#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

using namespace godot;

class PythonObject : public Resource {
	GDCLASS(PythonObject, Resource);

    friend class PythonRuntime;
private:
    PyObject *instance;
    String __name__;
    String __repr__;

    Variant call_internal(const Variant **p_args, GDExtensionInt p_arg_count);

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
		std::array<const Variant *, sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return call_internal(call_args.data(), variant_args.size());
	}

    Variant call(const Array &p_args = Array(), const Dictionary &p_kwargs = Dictionary());

    Ref<PythonObject> getattr(const String &);
    bool is_callable();
};
