#include "python_runtime.h"

#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/engine.hpp>

PyMODINIT_FUNC PyInit_gdextension(void);
PyMODINIT_FUNC PyInit_entry_point(void);
PyMODINIT_FUNC PyInit__godot_type_tuples(void);
PyMODINIT_FUNC PyInit_godot_types(void);

using namespace godot;

PythonRuntime *PythonRuntime::singleton = nullptr;

PythonRuntime::PythonRuntime() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
	initialized = false;
	interpreter_state = nullptr;
	main_thread_state = nullptr;
	main_thread_id = 0;

	internal::gdextension_interface_get_library_path(internal::library, &library_path);
}

void PythonRuntime::pre_initialize() {
	UtilityFunctions::print_verbose("Python: Pre-Initializing runtime...");

	PyPreConfig preconfig;
	PyPreConfig_InitIsolatedConfig(&preconfig);

	preconfig.utf8_mode = 1;

	PyStatus status = Py_PreInitialize(&preconfig);

	if (PyStatus_Exception(status)) {
		UtilityFunctions::push_error("Python: Pre-Initialization FAILED");
		Py_ExitStatusException(status);
	}
}

#define ERR_FAIL_PYSTATUS(status, label) if (PyStatus_Exception(status)) goto label
#define CHECK_PYSTATUS(status, ret) if (PyStatus_Exception(status)) return ret

#define SET_PYCONFIG_STRING(config_param, path) \
	status = PyConfig_SetString(config, config_param, path); \
	if (PyStatus_Exception(status)) return 1

#define APPEND_PYTHON_PATH(path) \
	status = PyWideStringList_Append(&config->module_search_paths, path); \
	if (PyStatus_Exception(status)) return 1

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

int PythonRuntime::set_config_paths(PyConfig *config) {
	PyStatus status;

	String exec_path = library_path;
	String exec_prefix = exec_path.get_base_dir();

	String bin_dir = exec_prefix.get_base_dir();
	String res_path = bin_dir.get_base_dir();

	UtilityFunctions::print_verbose("Python library name: " + exec_path);
	UtilityFunctions::print_verbose("Detected project folder: " + res_path);

	SET_PYCONFIG_STRING(&config->program_name, exec_path.wide_string());
	SET_PYCONFIG_STRING(&config->base_exec_prefix, exec_prefix.wide_string());
	SET_PYCONFIG_STRING(&config->base_prefix, res_path.wide_string());
	SET_PYCONFIG_STRING(&config->exec_prefix, exec_prefix.wide_string());
	SET_PYCONFIG_STRING(&config->executable, exec_path.wide_string());
	SET_PYCONFIG_STRING(&config->prefix, res_path.wide_string());

	String python_lib_path = res_path.path_join("python").path_join("windows").path_join("lib");

	APPEND_PYTHON_PATH(res_path.wide_string());
	APPEND_PYTHON_PATH(res_path.path_join("lib").wide_string());
	APPEND_PYTHON_PATH(python_lib_path.wide_string());
	APPEND_PYTHON_PATH(python_lib_path.path_join("site-packages").wide_string());

#ifdef GODOPY_LIB_PATH
	String godopy_lib_path = STR(GODOPY_LIB_PATH);
	APPEND_PYTHON_PATH(godopy_lib_path.wide_string());
#endif

	return 0;
}

void PythonRuntime::initialize() {
	pre_initialize();

	UtilityFunctions::print_verbose("Python: Initializing runtime...");

	PyStatus status;
	PyConfig config;

	PyImport_AppendInittab("gdextension", PyInit_gdextension);
	PyImport_AppendInittab("entry_point", PyInit_entry_point);
	PyImport_AppendInittab("_godot_type_tuples", PyInit__godot_type_tuples);
	PyImport_AppendInittab("godot_types", PyInit_godot_types);

	PyConfig_InitIsolatedConfig(&config);

	UtilityFunctions::print_verbose("Python: Configuring paths...");

	if (set_config_paths(&config) != 0) {
		goto fail;
	}

	config.site_import = 0;
	config.install_signal_handlers = 0;
	config.module_search_paths_set = 1;

	status = PyConfig_Read(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	UtilityFunctions::print_verbose("Python: Initializing the interpreter...");
	status = Py_InitializeFromConfig(&config);
	ERR_FAIL_PYSTATUS(status, fail);

	initialized = true;

	PyConfig_Clear(&config);

	interpreter_state = PyInterpreterState_Get();

	// Also releases the GIL
	// This seems to be important, otherwise callbacks from different threads create deadlocks
	main_thread_state = PyEval_SaveThread();

	return;

fail:
	PyConfig_Clear(&config);
	UtilityFunctions::push_error("Python: Initialization FAILED.");
	Py_ExitStatusException(status);
}

void PythonRuntime::run_simple_string(const String &p_string) {
	ERR_FAIL_COND(!initialized);
	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyRun_SimpleString(p_string.utf8());
	PyGILState_Release(gil_state);
}

Ref<PythonObject> PythonRuntime::import_module(const String &p_name) {
	Ref<PythonObject> module = memnew(PythonObject);
	module->set_name(p_name);

	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyObject *m = PyImport_ImportModule(p_name.utf8());

	if (PyErr_Occurred()) {
		PyObject *exc = PyErr_GetRaisedException();
		ERR_FAIL_NULL_V(exc, module);
		print_traceback(exc);
        Py_DECREF(exc);
	}

	ERR_FAIL_NULL_V(m, module);

    Py_INCREF(m);
    module->set_instance(m);
	PyObject *repr = PyObject_Repr(m);
    ERR_FAIL_NULL_V(repr, module);
    module->set_repr(String(repr));
	Py_DECREF(m);
	Py_DECREF(repr);
    PyGILState_Release(gil_state);

    return module;
}

void PythonRuntime::init_module(const String &p_name) {
	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyObject *m = PyImport_ImportModule(p_name.utf8());

	if (PyErr_Occurred()) {
		PyObject *exc = PyErr_GetRaisedException();
		ERR_FAIL_NULL(exc);
		print_traceback(exc);
        Py_DECREF(exc);
	}

	ERR_FAIL_NULL(m);
	Py_DECREF(m);
	PyGILState_Release(gil_state);
}

PythonObject *PythonRuntime::python_object_from_pyobject(PyObject *p_obj) {
	Ref<PythonObject> obj = memnew(PythonObject);

	PyGILState_STATE gil_state = PyGILState_Ensure();

	Py_INCREF(p_obj);
	obj->set_instance(p_obj);

	PyGILState_Release(gil_state);

	obj->reference();

	return obj.ptr();
}

void PythonRuntime::ensure_current_thread_state(bool setdefault) {
	PyThreadState *tstate;
	uint64_t thread_id = OS::get_singleton()->get_thread_caller_id();

	if (setdefault) {
		main_thread_id = thread_id;
		UtilityFunctions::print_verbose("Main thread id #", main_thread_id);
		PyGILState_STATE gil_state = PyGILState_Ensure();
		tstate = PyThreadState_Get();
		thread_states[thread_id] = tstate;
		PyGILState_Release(gil_state);

	} else if (thread_states.find(thread_id) == thread_states.end()) {
		// Python requires separate thread states for each C++ thread
		tstate = PyThreadState_New(interpreter_state);
		thread_states[thread_id] = tstate;

		UtilityFunctions::print_verbose("Set Python thread state for Godot thread #", thread_id);
	}
}

void PythonRuntime::finalize() {
	if (is_initialized()) {
		if (Py_IsInitialized()) {
			PyGILState_STATE gil_state = PyGILState_Ensure();
			for (auto it: thread_states) {
				if (it.first != main_thread_id) {
					PyThreadState_Clear(it.second);
					PyThreadState_Delete(it.second);
					thread_states[it.first] = nullptr;
				}
			}
			PyGILState_Release(gil_state);
			PyEval_RestoreThread(main_thread_state);

			main_thread_state = nullptr;
			interpreter_state = nullptr;

			Py_FinalizeEx();
		}

		initialized = false;
	}
}

PythonRuntime::~PythonRuntime() {
	ERR_FAIL_COND(singleton != this);
	singleton = nullptr;
}
