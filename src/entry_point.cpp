#include "python/python_runtime.h"
#include "python/python_object.h"

#include <binding.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

extern int python_initialize_level(ModuleInitializationLevel);
extern int python_deinitialize_level(ModuleInitializationLevel);

#define MINIMUM_INITIALIZATION_LEVEL MODULE_INITIALIZATION_LEVEL_SCENE

void initialize_level(ModuleInitializationLevel p_level) {
	if (p_level == MINIMUM_INITIALIZATION_LEVEL) {
		// The only GDExtension class defined at godot-cpp level
		// Simple PyObject* wrapper, provides getattr and call methods
		GDREGISTER_CLASS(PythonObject);

		// Initialize thread handling
		PythonRuntime::get_singleton()->ensure_current_thread_state(true);

		// Import core modules so we can safely call initialization functions from C++
		PythonRuntime::get_singleton()->init_module("godot_types");
		PythonRuntime::get_singleton()->init_module("entry_point");
		PythonRuntime::get_singleton()->init_module("gdextension");
	}

	if (p_level >= MINIMUM_INITIALIZATION_LEVEL) {
		python_initialize_level(p_level);
	}
}

void deinitialize_level(ModuleInitializationLevel p_level) {
	if (p_level >= MINIMUM_INITIALIZATION_LEVEL) {
		python_deinitialize_level(p_level);
	}
}

extern "C" GDExtensionBool GDE_EXPORT godopy_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_level);
	init_obj.register_terminator(deinitialize_level);
	init_obj.set_minimum_library_initialization_level(MINIMUM_INITIALIZATION_LEVEL);

	return init_obj.init();
}
