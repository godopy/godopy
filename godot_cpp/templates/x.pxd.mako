% for cimp, symbols in cimports.items():
from ${cimp} cimport ${', '.join(symbols)}
% endfor

% for name, cls in registered_classes.items():
cdef extern from "${basename}__w.hpp" namespace "godot":
    cdef cppclass ${cls['name']}(${cls['base']}):
    % if 'attrs' in cls or 'props' in cls:
        % if 'attrs' in cls:
            % for attr, attrtype in cls['attrs'].items():
        ${attrtype} ${attr}
            % endfor
        % endif
        % if 'props' in cls:
            % for prop, proptype in cls['props'].items():
        ${proptype} ${prop}
            % endfor
        % endif
    % else:
        pass
    % endif

    % for mname, mdef in cls['methods'].items():
cdef public ${mdef['return_type']} ${mname}(
        % for arg, argtype in mdef['args'].items():
    ${argtype} ${arg}${')' if loop.last else ','}
        % endfor

    % endfor

% endfor
