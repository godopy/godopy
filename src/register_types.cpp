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

static ModuleInitializationLevel MIN_LEVEL = MODULE_INITIALIZATION_LEVEL_SCENE;

void initialize_godopy_types(ModuleInitializationLevel p_level)
{
	if (p_level == MIN_LEVEL) {
		runtime = memnew(PythonRuntime);
		runtime->pre_initialize();
		runtime->initialize();
	}

	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		ClassDB::register_class<Python>();
		python = memnew(Python);
		Engine::get_singleton()->register_singleton("Python", Python::get_singleton());

		ClassDB::register_class<PythonObject>();
	}


	if (p_level >= MIN_LEVEL) {
		Ref<PythonObject> mod = PythonRuntime::get_singleton()->import_module("_godot");
		Ref<PythonObject> py_init_func = mod->getattr("initialize_godopy_types");
		py_init_func->call_one_arg(Variant(p_level));
		py_init_func.unref();
		mod.unref();
	}
}

void uninitialize_godopy_types(ModuleInitializationLevel p_level) {
	if (p_level >= MIN_LEVEL) {
		Ref<PythonObject> mod = PythonRuntime::get_singleton()->import_module("_godot");
		Ref<PythonObject> py_term_func = mod->getattr("uninitialize_godopy_types");
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
	GDExtensionBool GDE_EXPORT godopy_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization)
	{
		GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
		init_obj.register_initializer(initialize_godopy_types);
		init_obj.register_terminator(uninitialize_godopy_types);
		init_obj.set_minimum_library_initialization_level(MIN_LEVEL);

		return init_obj.init();
	}
}