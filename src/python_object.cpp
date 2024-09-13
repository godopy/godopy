#include "python_object.h"

using namespace godot;

PythonModule::PythonModule() {
    obj = NULL;
}

PythonModule::~PythonModule() {
    Py_XDECREF(obj);
}

PythonModule *PythonModule::import(const String &p_name) {
    PythonModule *module = memnew(PythonModule);
    module->obj = PyImport_ImportModule(p_name.utf8());
    // TODO: Check for nullptr and print error
    module->name = p_name;
    Py_XINCREF(module->obj);

    return module;
}

void PythonModule::_bind_methods() {
	ClassDB::bind_static_method("PythonModule", D_METHOD("import"), &PythonModule::import);
}
