# Generated by PyGodot binding generator
<%!
    from godot_tools.binding_generator import python_module_name, remove_nested_type_prefix, CYTHON_ONLY_ESCAPES
    enum_values = set()

    reversed_escapes = {v: k for k, v in CYTHON_ONLY_ESCAPES.items()}

    def clean_value_name(value_name):
        enum_values.add(value_name)
        return remove_nested_type_prefix(value_name)

    def escape_real_method_name(name):
        if name in reversed_escapes:
            return '%s "%s" ' % (name, reversed_escapes[name])
        else:
            return name
%>
# from godot_headers.gdnative_api cimport *
from ..core.cpp_types cimport *
% for class_name, class_def, includes, forwards, methods in classes:


cdef extern from "${class_name}.hpp" namespace "godot" nogil:
    cdef cppclass ${class_name}(${class_def['base_class'] or '__cpp_internal_Wrapped'}):
% for enum in class_def['enums']:
        enum ${enum['name'].lstrip('_')}:
    % for value_name, value in enum['values'].items():
            ${clean_value_name(value_name)} = ${value}
    % endfor

% endfor
% for name, value in ((k, v) for (k, v) in class_def['constants'].items() if k not in enum_values):
        enum:
            ${name} = ${value}

% endfor
% for method_name, return_type, pxd_signature, signature, args in methods:
        ${return_type}${escape_real_method_name(method_name)}(${signature})
% endfor
% if class_def['singleton']:

        @staticmethod
        ${class_name} *get_singleton()
% elif class_def['instanciable']:

        ${class_name}() except +
% elif not class_def['enums'] and not class_def['constants'] and not methods:
        pass
% endif
% endfor
