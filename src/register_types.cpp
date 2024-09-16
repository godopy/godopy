#include "register_types.h"
#include "python_runtime.h"
#include "python_object.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/engine.hpp>

using namespace godot;

static PythonRuntime *runtime;
static Python *python;

void initialize_godopy_types(ModuleInitializationLevel p_level)
{
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	runtime = memnew(PythonRuntime);
	runtime->pre_initialize();
	runtime->initialize();

	ClassDB::register_class<Python>();
	python = memnew(Python);
	Engine::get_singleton()->register_singleton("Python", Python::get_singleton());

	ClassDB::register_class<PythonObject>();

	PythonObject *mod = PythonRuntime::get_singleton()->import_module("_godot");
	PythonObject *py_init_func = mod->getattr("initialize_types");
	py_init_func->call();
}

void terminate_godopy_types(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	PythonObject *mod = PythonRuntime::get_singleton()->import_module("_godot");
	PythonObject *py_term_func = mod->getattr("terminate_types");
	py_term_func->call();

	Engine::get_singleton()->unregister_singleton("Python");
	if (python) {
		memdelete(python);
	}
	if (runtime) {
		memdelete(runtime);
	}
}

extern "C"
{
	// Initialization
	GDExtensionBool GDE_EXPORT godopy_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization)
	{
		GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
		init_obj.register_initializer(initialize_godopy_types);
		init_obj.register_terminator(terminate_godopy_types);
		init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

		return init_obj.init();
	}
}