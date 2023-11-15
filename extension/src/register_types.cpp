#include "register_types.h"
#include "python_runtime.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

static PythonRuntime *runtime;
static Python *python;

void gdextension_initialize(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		runtime = memnew(PythonRuntime);
        runtime->pre_initialize();
        runtime->initialize();

		runtime->run_simple_string("import sys; print(sys.path); print('OK')");

		ClassDB::register_class<Python>();

        python = memnew(Python);

		Engine::get_singleton()->register_singleton("Python", Python::get_singleton());
    }
}

void gdextension_terminate(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
		
	} else if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
        Engine::get_singleton()->unregister_singleton("Python");
        if (python) {
            memdelete(python);
        }

		if (runtime) {
		    memdelete(runtime);
        }
    }
}

extern "C" {
	GDExtensionBool GDE_EXPORT gdextension_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
		GDExtensionInitialization *r_initialization
    ) {
		godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

		init_obj.register_initializer(gdextension_initialize);
		init_obj.register_terminator(gdextension_terminate);
		init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SERVERS);

		return init_obj.init();
	}
}
