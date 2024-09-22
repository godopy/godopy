#include "python/python_runtime.h"
#include "python/python_object.h"

#include <gdextension_interface.h>
#include <binding.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

static Python *python;

void initialize_level(ModuleInitializationLevel p_level)
{
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		python = memnew(Python);
		Engine::get_singleton()->register_singleton("Python", Python::get_singleton());
	}

	if (p_level == MODULE_INITIALIZATION_LEVEL_CORE) {
		ClassDB::register_class<Python>();
		ClassDB::register_class<PythonObject>();
	}

	if (p_level >= MODULE_INITIALIZATION_LEVEL_SCENE) {
		Ref<PythonObject> mod = PythonRuntime::get_singleton()->import_module("_godopy_core");
		Ref<PythonObject> py_init_func = mod->getattr("initialize_level");
		py_init_func->call_one_arg(Variant(p_level));
		py_init_func.unref();
		mod.unref();
	}
}

void deinitialize_level(ModuleInitializationLevel p_level) {
	if (p_level >= MODULE_INITIALIZATION_LEVEL_CORE) {
		Ref<PythonObject> mod = PythonRuntime::get_singleton()->import_module("_godopy_core");
		Ref<PythonObject> py_term_func = mod->getattr("deinitialize_level");
		py_term_func->call_one_arg(Variant(p_level));
		py_term_func.unref();
		mod.unref();
	}

	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		Engine::get_singleton()->unregister_singleton("Python");
		if (python) {
			memdelete(python);
		}
	}
}

extern "C"
{
	// Initialization
	GDExtensionBool GDE_EXPORT godopy_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {

		GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

		init_obj.register_initializer(initialize_level);
		init_obj.register_terminator(deinitialize_level);

		return init_obj.init();
	}
}
