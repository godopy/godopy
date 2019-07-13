% for cimp, symbols in registry.cimports.items():
from ${cimp} cimport ${', '.join(symbols)}
% endfor

% for cls in registry.classes:
cdef extern from "${basename}__wrap.hpp" namespace "godot":
    cdef cppclass ${cls.name}(${cls.base}):
    % if len(cls.attrs):
        % for attr, attrtype in cls.attrs.items():
        ${attrtype} ${attr}
        % endfor
    % else:
        pass
    % endif

    % for mname, spec in cls.methods.items():
cdef public ${spec.annotations['return']} ${mname}(
        % for arg in spec.args:
    ${spec.annotations[arg]} ${arg}${')' if loop.last else ','}
        % endfor

    % endfor

% endfor
