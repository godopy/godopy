import os
from mako.template import Template

CORE_TYPES = frozenset(['Array', 'Basis', 'Color', 'Dictionary', 'Error', 'NodePath', 'Plane',
    'PoolByteArray', 'PoolIntArray', 'PoolRealArray', 'PoolStringArray', 'PoolVector2Array', 'PoolVector3Array',
    'PoolColorArray', 'PoolIntArray', 'PoolRealArray', 'Quat', 'Rect2', 'AABB', 'RID', 'String', 'Transform',
    'Transform2D', 'Variant', 'Vector2', 'Vector3'])

PRIMITIVE_TYPES = frozenset(['int', 'bool', 'real', 'float', 'void'])

def generate(root_dir, echo=print):
    from ..headers import api

    templates_dir = os.path.join(root_dir, 'pygodot', 'binding_generator', 'templates')
    icalls_header_path = os.path.join(root_dir, 'include', 'pygen', '__py_icalls.hpp')

    icalls = set()

    header_contexts = []
    impl_contexts = []

    for name, cls in api.CLASSES.items():
        used_types = _get_used_types(cls)
        class_name = strip_name(name)

        for method in cls['methods'].values():
            if method['has_varargs']:
                continue
            args = tuple(get_icall_type_name(arg['type']) for arg in method['arguments'])
            ret = get_icall_type_name(method['return_type'])
            icalls.add((ret, args))

    prepared_icalls = []

    for ret_type, args in icalls:
        sig = [
            get_icall_return_type(ret_type), ' ',
            get_icall_name(ret_type, args), '(',
            'godot_method_bind *mb, ', 'godot_object *o'
        ]
        for i, arg in enumerate(args):
            sig.append(', const ')
            if is_core_type(arg):
                sig.append('godot::' +arg + '& ')
            elif arg == 'int':
                sig.append('int64_t ')
            elif arg == 'float':
                sig.append('double ')
            elif is_primitive(arg):
                sig.append(arg + ' ')
            else:
                sig.append('godot_object *')

            sig.append(f'arg{i}')
        sig.append(')')

        prepared_icalls.append((ret_type, args, ''.join(sig), ret_type != 'void'))

    icalls_header_template = Template(filename=os.path.join(templates_dir, '__py_icalls.hpp.mako'))
    icalls_header = icalls_header_template.render(icalls=prepared_icalls)

    with open(icalls_header_path, 'w', encoding='utf-8') as fp:
        fp.write(icalls_header)


def get_icall_return_type(t):
    if is_class_type(t):
        return '__pygodot___Wrapped *'
    elif t == 'int':
        return 'int64_t'
    elif t == 'float' or t == 'real':
        return 'double'
    elif is_primitive(t):
        return t
    else:
        return f'godot::{t}'


def get_icall_name(ret_type, args):
    name = "___pygodot_icall_"
    name += strip_name(ret_type)
    for arg in args:
        name += "_" + strip_name(arg)

    return name

def get_icall_type_name(name):
    if name.startswith('enum'):
        return 'int'
    if is_class_type(name):
        return 'Object'
    return name

def _get_used_types(cls):
    classes = set()
    for method in cls['methods'].values():
        if is_class_type(method['return_type']) and not (method['return_type'] in classes):
            classes.add(method['return_type'])

        for arg in method['arguments']:
            if is_class_type(arg['type']) and not (arg['type'] in classes):
                classes.add(arg['type'])
    return classes

def is_class_type(name):
    return not name in CORE_TYPES and not name in PRIMITIVE_TYPES

def is_core_type(name):
    return name in CORE_TYPES

def is_primitive(name):
    return name in PRIMITIVE_TYPES

def strip_name(name):
    if len(name) == 0:
        return name
    if name[0] == '_':
        return name[1:]
    return name
