#include "python_module.h"

using namespace godot;

PythonModule::PythonModule() {
    obj = NULL;
}

PythonModule::~PythonModule() {
    Py_XDECREF(obj);
}

PythonModule *PythonModule::import_module(const String &p_name) {
    PythonModule *module = memnew(PythonModule);
    module->obj = PyImport_ImportModule(p_name.utf8());
    module->name = p_name;
    Py_XINCREF(module->obj);

    return module;
}

void PythonModule::_bind_methods() {
	ClassDB::bind_static_method("PythonModule", D_METHOD("import_module"), &PythonModule::import_module);
}
