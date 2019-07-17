#ifndef ${basename.upper()}__WRAP_HPP
#define ${basename.upper()}__WRAP_HPP

#include <Godot.hpp>
% for cls in registry.classes:
#include <${cls.base}.hpp>
% endfor

namespace godot {

% for cls in registry.classes:
class ${cls.name} : public ${cls.base} {
	GODOT_CLASS(${cls.name}, ${cls.base})

public:
	% if len(cls.attrs):
		% for attr, attrtype in cls.attrs.items():
	${attrtype} ${attr};
		% endfor
	% endif

	static void _register_methods();

	${cls.name}() {}
	~${cls.name}() {}
	% for mname, spec in cls.methods.items():

	${spec.annotations['return']} ${mname}(
		% for arg in spec.args:
			% if not loop.first:
		${spec.annotations[arg]} ${arg}${'' if loop.last else ','}
			% endif
		% endfor
	);
	% endfor
};
% endfor

}
#endif
