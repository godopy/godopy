#include "GodotGlobal.hpp"
#include "PythonGlobal.hpp"

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

// TODO: #include these defs
extern "C" __pygodot___Wrapped *_create_wrapper(godot_object *, size_t);
extern "C" void __init_cython_method_bindings(void);
extern "C" void __register_cython_types(void);
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

void PyGodot::nativescript_init(void *handle, bool init_cython) {
	godot::_RegisterState::nativescript_handle = handle;

	godot_instance_binding_functions binding_funcs = {};
	binding_funcs.alloc_instance_binding_data = wrapper_create;
	binding_funcs.free_instance_binding_data = wrapper_destroy;
	binding_funcs.refcount_incremented_instance_binding = wrapper_incref;
	binding_funcs.refcount_decremented_instance_binding = wrapper_decref;

  godot::_RegisterState::cython_language_index = godot::nativescript_1_1_api->godot_nativescript_register_instance_binding_data_functions(binding_funcs);
	godot::_RegisterState::python_language_index = godot::nativescript_1_1_api->godot_nativescript_register_instance_binding_data_functions(binding_funcs);

  if (init_cython) {
    __register_cython_types();
    __init_cython_method_bindings();
  }

	__register_python_types();
	__init_python_method_bindings();
}

void PyGodot::nativescript_terminate(void *handle) {
	godot::nativescript_1_1_api->godot_nativescript_unregister_instance_binding_data_functions(godot::_RegisterState::python_language_index);
}

} // namespace pygodot
