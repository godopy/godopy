#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"
#include "pygodot/pyscript.h"

#include <wchar.h>

static PyTypeObject _WrappedType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_pygodot._Wrapped",
    .tp_doc = "",
    .tp_basicsize = sizeof(__pygodot___Wrapped),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};

// The name should be the same as the binary's name as it makes this GDNative library also importable by Python
static PyModuleDef _pygodotmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_pygodot",
    .m_doc = "PyGodot GDNative extension",
    .m_size = -1,
};

extern "C" __pygodot___Wrapped *_create_wrapper(godot_object *, size_t);
extern "C" void __init_python_method_bindings(void);
extern "C" void __register_python_types(void);

PyMODINIT_FUNC PyInit__pygodot(void) {
  PyObject *m;
  if (PyType_Ready(&_WrappedType) < 0)
    return NULL;

  m = PyModule_Create(&_pygodotmodule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&_WrappedType);
  PyModule_AddObject(m, "_Wrapped", (PyObject *) &_WrappedType);
  return m;
}

static GDCALLINGCONV void *wrapper_create(void *data, const void *type_tag, godot_object *instance) {
	// XXX: call PyObject_New directly?
	__pygodot___Wrapped *wrapper_obj = _create_wrapper(instance, (size_t)type_tag);

	return (void *)wrapper_obj;
}

static GDCALLINGCONV void wrapper_incref(void *data, void *wrapper) {
	if (wrapper)
		Py_INCREF((PyObject *)wrapper);
}

static GDCALLINGCONV bool wrapper_decref(void *data, void *wrapper) {
	if (wrapper)
		Py_DECREF((PyObject *)wrapper);
	return (bool) !wrapper; // FIXME
}

static GDCALLINGCONV void wrapper_destroy(void *data, void *wrapper) {
	if (wrapper)
		Py_DECREF((PyObject *)wrapper);
}

namespace pygodot {


static const char *pyscript_registered_extensions[] = {"py", "pyw", NULL};
static const char *pyscript_reserved_words[] = {"False", "None", "True", "and", "as", "assert", "break", "class",
  "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is",
  "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", NULL};
static const char *pyscript_comment_delimeters[] = {"#", NULL};
static const char *pyscript_string_delimeters[] = {"\" \"", "' '", NULL};

static const godot_pluginscript_instance_desc pyscript_instance = {
  .init = pyscript_instance_init,
  .finish = pyscript_instance_finish,

  .set_prop = pyscript_instance_set_prop,
  .get_prop = pyscript_instance_get_prop,
  .call_method = pyscript_instance_call_method,
  .notification = pyscript_instance_notification,

  .get_rpc_mode = NULL,
  .get_rset_mode = NULL,

  .refcount_incremented = NULL, // pyscript_instance_refcount_incremented,
  .refcount_decremented = NULL, // pyscript_instance_refcount_decremented
};

static const godot_pluginscript_script_desc pyscript = {
  .init = pyscript_init,
  .finish = pyscript_finish,
  .instance_desc = pyscript_instance
};

static godot_bool pyscript_validate(godot_pluginscript_language_data *p_data,
  const godot_string *p_script, int *r_line_error, int *r_col_error, godot_string *r_test_error,
  const godot_string *p_path, godot_pool_string_array *r_functions) { return true; }

static const godot_pluginscript_language_desc pyscript_language = {
  .name = "Python",
  .type = "Python", // Godot does not use this field at the moment, always returns "PluginScript"
  .extension = "py",
  .recognized_extensions = pyscript_registered_extensions,
  .init = pyscript_language_init,
  .finish = pyscript_language_finish,
  .reserved_words = pyscript_reserved_words,
  .comment_delimiters = pyscript_comment_delimeters,
  .string_delimiters = pyscript_string_delimeters,
  .has_named_classes = false, // XXX check
  .supports_builtin_mode = false, // XXX check

  .get_template_source_code = pyscript_get_template_source_code,
  .validate = pyscript_validate,
  .find_function = NULL,
  .make_function = NULL,
  .complete_code = NULL,
  .auto_indent_code = NULL,

  .add_global_constant = pyscript_add_global_constant,
  .debug_get_error = NULL,
  .debug_get_stack_level_count = NULL,
  .debug_get_stack_level_line = NULL,
  .debug_get_stack_level_function = NULL,
  .debug_get_stack_level_source = NULL,
  .debug_get_stack_level_locals = NULL,
  .debug_get_stack_level_members = NULL,
  .debug_get_globals = NULL,
  .debug_parse_stack_level_expression = NULL,

  .get_public_functions = NULL,
  .get_public_constants = NULL,

  .profiling_start = NULL,
  .profiling_stop = NULL,
  .profiling_get_accumulated_data = NULL,
  .profiling_get_frame_data = NULL,
  .profiling_frame = NULL,

  .script_desc = pyscript
};

wchar_t *pythonpath = nullptr;

void PyGodot::set_pythonpath(godot_gdnative_init_options *options) {
	const godot_gdnative_core_api_struct *api = options->api_struct;

	godot_string dir = api->godot_string_get_base_dir(options->active_library_path);
  godot_string file = api->godot_string_get_file(options->active_library_path);
	godot_int dirsize = api->godot_string_length(&dir);
  godot_int filesize = api->godot_string_length(&file);

	pythonpath = (wchar_t *)PyMem_RawMalloc((dirsize + 1 + filesize + 5) * sizeof(wchar_t));
	wcsncpy(pythonpath, api->godot_string_wide_str(&dir), dirsize);
  wcsncpy(pythonpath + dirsize, L"/", 1);
  wcsncpy(pythonpath + dirsize + 1, api->godot_string_wide_str(&file), filesize);
	wcsncpy(pythonpath + dirsize + 1 + filesize, L".env", 5);

	api->godot_string_destroy(&dir);
  api->godot_string_destroy(&file);
}

void PyGodot::python_init() {
	if (!pythonpath) {
		printf("Could not initialize Python interpreter:\n");
		printf("Python path was not defined!\n");

		return;
	}

  Py_NoUserSiteDirectory = 1;

#ifdef PYGODOT_EXPORT
  Py_NoSiteFlag = 1;
	Py_IgnoreEnvironmentFlag = 1;
#endif

	Py_SetProgramName(L"godot");
	Py_SetPythonHome(pythonpath);

	// Initialize interpreter but skip initialization registration of signal handlers
	Py_InitializeEx(0);

	PyObject *mod = PyImport_ImportModule("pygodot");
  if (mod != NULL) {
    Py_DECREF(mod);

    printf("Python %s\n\n", Py_GetVersion());
  } else {
    PyErr_Print();
  }
}

void PyGodot::python_terminate() {
	if (Py_IsInitialized()) {
		Py_FinalizeEx();

		if (pythonpath)
			PyMem_RawFree((void *)pythonpath);
	}
}

/***
 * ORDER IS IMPORTANT!
 * 1. PyImport_AppendInittab for "_core", "Godot", "Bindings" and all user extentions
 * 2. pygodot::PyGodot::python_init();
 * 3. PyImport_ImportModule for "Godot", "Bindings" and all user extentions
 * 4. Only after all previous steps: pygodot::PyGodot::nativescript_init(handle);
 * 5. Register user classes
 *
 * Both godot::Godot::nativescript_init and pygodot::PyGodot::nativescript_init can be used in the same program
 ***/
void PyGodot::nativescript_init(void *handle) {
	godot::_RegisterState::nativescript_handle = handle;

	godot_instance_binding_functions binding_funcs = {};
	binding_funcs.alloc_instance_binding_data = wrapper_create;
	binding_funcs.free_instance_binding_data = wrapper_destroy;
	binding_funcs.refcount_incremented_instance_binding = wrapper_incref;
	binding_funcs.refcount_decremented_instance_binding = wrapper_decref;

	godot::_RegisterState::python_language_index = godot::nativescript_1_1_api->godot_nativescript_register_instance_binding_data_functions(binding_funcs);

	__register_python_types();
	__init_python_method_bindings();
}

void PyGodot::nativescript_terminate(void *handle) {
	godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::python_language_index);
}

void PyGodot::register_pyscript_language() {
  godot::pluginscript_api->godot_pluginscript_register_language(&pyscript_language);
}

} // namespace pygodot
