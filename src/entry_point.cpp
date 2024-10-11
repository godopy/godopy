#include "python/python_runtime.h"
#include "python/python_object.h"

#include <gdextension_interface.h>
#include <binding.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

#define MINIMUM_INITIALIZATION_LEVEL MODULE_INITIALIZATION_LEVEL_SCENE

void initialize_level(ModuleInitializationLevel p_level) {
	if (p_level == MINIMUM_INITIALIZATION_LEVEL) {
		// The only GDExtension class defined at godot-cpp level
		// Simple PyObject* wrapper, provides getattr and call methods
		GDREGISTER_CLASS(PythonObject);
	}

	if (p_level >= MINIMUM_INITIALIZATION_LEVEL) {
		Ref<PythonObject> mod = PythonRuntime::get_singleton()->import_module("entry_point");
		Ref<PythonObject> py_init_func = mod->getattr("initialize_level");
		py_init_func->call_one_arg(Variant(p_level));
		py_init_func.unref();
		mod.unref();
	}
}

void deinitialize_level(ModuleInitializationLevel p_level) {
	if (p_level >= MINIMUM_INITIALIZATION_LEVEL) {
		Ref<PythonObject> mod = PythonRuntime::get_singleton()->import_module("entry_point");
		Ref<PythonObject> py_term_func = mod->getattr("deinitialize_level");
		py_term_func->call_one_arg(Variant(p_level));
		py_term_func.unref();
		mod.unref();
	}
}

extern "C"
{
	// Initialization
	GDExtensionBool GDE_EXPORT godopy_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {

		GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

		init_obj.register_initializer(initialize_level);
		init_obj.register_terminator(deinitialize_level);
		init_obj.set_minimum_library_initialization_level(MINIMUM_INITIALIZATION_LEVEL);

		return init_obj.init();
	}
}
