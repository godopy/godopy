# Generated by PyGodot binding generator
<%!
    from pygodot.cli.binding_generator import (
        python_module_name, remove_nested_type_prefix, CORE_TYPES, SPECIAL_ESCAPES, clean_signature
    )

    enum_values = set()

    def clean_value_name(value_name):
        enum_values.add(value_name)
        return remove_nested_type_prefix(value_name)
%>
from ..core_types cimport *

cdef __register_types()
cdef __init_method_bindings()
% for class_name, class_def, includes, forwards, methods in classes:

    % if methods:
cdef struct __${class_name}__method_bindings:
        % for method_name, method, return_type, pxd_signature, signature, args, return_stmt, init_args in methods:
    godot_method_bind *mb_${method_name}
        % endfor
    % endif

cdef class ${class_name}(${class_def['base_class'] or '_Wrapped'}):
    @staticmethod
    cdef __init_method_bindings()
    % if class_def['singleton']:

    @staticmethod
    cdef object get_singleton()
    % endif
    % if class_def['instanciable']:

    @staticmethod
    cdef ${class_name} _new()
    % endif

    % for method_name, method, return_type, pxd_signature, signature, args, return_stmt, init_args in methods:
    % if method['__func_type'] == 'cdef':
    cdef ${return_type}${method_name}(self${', ' if pxd_signature else ''}${clean_signature(pxd_signature, class_name)})
    % endif
    % endfor

    % for enum in class_def['enums']:
cdef enum ${class_name}${enum['name'].lstrip('_')}:
        % for value_name, value in enum['values'].items():
    ${python_module_name(class_name).upper()}_${clean_value_name(value_name)} = ${value}
        % endfor

    % endfor
    % for name, value in ((k, v) for (k, v) in class_def['constants'].items() if k not in enum_values):
cdef int ${python_module_name(class_name).upper()}_${name} = ${value}
    % endfor
% endfor
