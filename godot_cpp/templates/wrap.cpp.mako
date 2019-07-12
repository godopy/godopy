#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "${basename}__wrap.hpp"
#include "${basename}__impl.h"

using namespace godot;
% for name, cls in registered_classes.items():

void ${name}::_register_methods() {
	PyImport_AppendInittab("${basename}", PyInit_${basename}__impl);
	% for method in cls['methods']:
		% if method != '_init':
	register_method("${method}", &${name}::${method});
		% endif
	% endfor
}

% for mname, mdef in cls['methods'].items():
	% if mname == '_init':
void GDExample::_init() {
	PyObject *mod = PyImport_ImportModule("${basename}");

  if (mod != NULL) {
    Py_DECREF(mod);
    ${name} *self = this;
    ${method_map['_init']}(self);
  } else {
    PyErr_Print();
  }
}
	% else:

${mdef['return_type']} ${name}::${mname}(
	% for arg, argtype in mdef['args'].items():
		% if not loop.first:
	${argtype} ${arg}${'' if loop.last else ','}
		% endif
	% endfor
) {
	${name} *self = this;
	${method_map[mname]}(
	% for arg in mdef['args']:
		${arg}${'' if loop.last else ','}
	% endfor
	);
}
	% endif
% endfor
% endfor
