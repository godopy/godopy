#ifndef ${basename.upper()}__WRAP_HPP
#define ${basename.upper()}__WRAP_HPP

#include <Godot.hpp>
% for cls in registered_classes.values():
#include <${cls['base']}.hpp>
% endfor

namespace godot {

% for name, cls in registered_classes.items():
class ${cls['name']} : public ${cls['base']} {
	GODOT_CLASS(${cls['name']}, ${cls['base']})

public:
	% if 'attrs' in cls:
		% for attr, attrtype in cls['attrs'].items():
	${attrtype} ${attr};
		% endfor
	% endif
	% if 'props' in cls:
		% for prop, proptype in cls['props'].items():
	${proptype} ${prop};
		% endfor
	% endif

	static void _register_methods();

	${cls['name']}() {}
	~${cls['name']}() {}
	% for mname, mdef in cls['methods'].items():

	${mdef['return_type']} ${mname}(
		% for arg, argtype in mdef['args'].items():
			% if not loop.first:
		${argtype} ${arg}${'' if loop.last else ','}
			% endif
		% endfor
	);
	% endfor
};
% endfor

}
#endif
