#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "${basename}__wrap.hpp"
#include "${basename}__impl.h"

using namespace godot;
% for cls in registry.classes:

void ${cls.name}::_register_methods() {
	PyImport_AppendInittab("${basename}", PyInit_${basename}__impl);
	% for pmethod in cls.public_methods:
		% if pmethod != '_init':
	register_method("${pmethod}", &${cls.name}::${pmethod});
		% endif
	% endfor
  % for pname, pdef in cls.props.items():
  register_property<${cls.name}, ${pdef['type']}>("${pname}", &${cls.name}::${pname}, ${pdef['value']});
  % endfor
  % for pname, pdef in cls.getset_props.items():
  register_property<${cls.name}, ${pdef['type']}>("${pname}", &${cls.name}::${pdef['setter']}, &${cls.name}::${pdef['getter']}, ${pdef['value']});
  % endfor

  % for signame, sigargs in cls.signals.items():
  register_signal<${cls.name}>((char *)"${signame}",
    % for key, value in sigargs.items():
    "${key}", GODOT_VARIANT_TYPE_${value.upper()}${'' if loop.last else ','}
    % endfor
  );
  % endfor
}

% for mname, spec in cls.methods.items():
	% if mname == '_init':
void GDExample::_init() {
	PyObject *mod = PyImport_ImportModule("${basename}");

  if (mod != NULL) {
    Py_DECREF(mod);
    ${cls.name} *self = this;
    ${method_map['_init']}(self);
  } else {
    PyErr_Print();
  }
}
	% else:

${spec.annotations['return']} ${cls.name}::${mname}(
	% for arg in spec.args:
		% if not loop.first:
	${spec.annotations[arg]} ${arg}${'' if loop.last else ','}
		% endif
	% endfor
) {
	${cls.name} *self = this;
	${method_map[mname]}(
	% for arg in spec.args:
		${arg}${'' if loop.last else ','}
	% endfor
	);
}
	% endif
% endfor
% endfor
