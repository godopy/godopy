import os
import re
import json
from mako.template import Template

CORE_TYPES = frozenset([
    'Array', 'Basis', 'Color', 'Dictionary', 'Error', 'NodePath', 'Plane',
    'PoolByteArray', 'PoolIntArray', 'PoolRealArray', 'PoolStringArray', 'PoolVector2Array', 'PoolVector3Array',
    'PoolColorArray', 'PoolIntArray', 'PoolRealArray', 'Quat', 'Rect2', 'AABB', 'RID', 'String', 'Transform',
    'Transform2D', 'Variant', 'Vector2', 'Vector3'
])

PRIMITIVE_TYPES = frozenset(['int', 'bool', 'real', 'float', 'void'])

CPP_ESCAPES = {
    'class':    '_class',
    'char':     '_char',
    'short':    '_short',
    'bool':     '_bool',
    'int':      '_int',
    'default':  '_default',
    'case':     '_case',
    'switch':   '_switch',
    'export':   '_export',
    'template': '_template',
    'new':      'new_',
    'operator': '_operator',
    'typename': '_typename',
}

CYTHON_ONLY_ESCAPES = {
    'from':     'from_',
    'pass':     'pass_',
    'raise':    'raise_',
    'global':   'global_',
}

CYTHON_ESCAPES = {**CPP_ESCAPES, **CYTHON_ONLY_ESCAPES}

reference_types = set()


def generate(root_dir, echo=print):
    with open(os.path.join(root_dir, 'godot_headers', 'api.json'), encoding='utf-8') as fp:
        classes = json.load(fp)

    for c in classes:
        if c['is_reference']:
            reference_types.add(strip_name(c['name']))

    node_types = set(['Node', 'TreeItem'])
    resource_types = set(['Resource', 'TriangleMesh'])
    engine_types = set()

    for i in range(8):
        for c in classes:
            if strip_name(c['base_class']) not in node_types:
                continue
            class_name = strip_name(c['name'])
            node_types.add(class_name)

    for i in range(8):
        for c in classes:
            if strip_name(c['base_class']) not in resource_types:
                continue
            class_name = strip_name(c['name'])
            resource_types.add(class_name)

    for c in classes:
        class_name = strip_name(c['name'])
        if class_name not in node_types and class_name not in resource_types:
            engine_types.add(class_name)

    print(len(node_types), len(resource_types), len(engine_types))

    templates_dir = os.path.join(root_dir, 'pygodot', 'binding_generator', 'templates')
    icalls_header_path = os.path.join(root_dir, 'include', 'pygen', '__py_icalls.hpp')

    class_contexts = []

    for class_def in classes:
        used_types = _get_used_types(class_def)
        class_name = class_def.pop('name').lstrip('_')

        class_contexts.append(generate_class_context(class_name, class_def, used_types, node_types, resource_types))

    prepared_icalls = generate_icalls_context(classes)

    icalls_header_template = Template(filename=os.path.join(templates_dir, '__py_icalls.hpp.mako'))
    icalls_header = icalls_header_template.render(icalls=prepared_icalls)

    with open(icalls_header_path, 'w', encoding='utf-8') as fp:
        fp.write(icalls_header)

    cpp_package_template = Template(filename=os.path.join(templates_dir, 'ccp_package.pxd.mako'))

    for cpp_package, _types in (('nodes', node_types), ('resources', resource_types), ('engine', engine_types)):
        cpp_path = os.path.join(root_dir, 'pygodot', 'cpp', cpp_package, '__init__.py')
        cpp_source = cpp_package_template.render(
            package=cpp_package,
            classes=[c for c in class_contexts if c[0] in _types]
        )
        with open(cpp_path, 'w', encoding='utf-8') as fp:
            fp.write(cpp_source)

        for class_name, class_def, includes, forwards, prepared_methods in class_contexts:
            if class_name not in _types:
                continue
            mod_name = '%s.pxd' % python_module_name(class_name)
            ccnode_path = os.path.join(root_dir, 'pygodot', 'cpp', cpp_package, mod_name)
            ccnode_template = Template(filename=os.path.join(templates_dir, 'cpp_binding.pxd.mako'))
            ccnode_source = ccnode_template.render(
                class_name=class_name,
                class_def=class_def,
                includes=includes,
                forwards=forwards,
                methods=prepared_methods,
                package=cpp_package
            )

            with open(ccnode_path, 'w', encoding='utf-8') as fp:
                fp.write(ccnode_source)


def generate_icalls_context(classes):
    icalls = set()

    for c in classes:
        for method in c['methods']:
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
                sig.append('godot::' + arg + '& ')
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

    return prepared_icalls


def make_gdnative_type(t):
    if is_enum(t):
        return '%s ' % remove_enum_prefix(t).replace('::', '.')
    elif is_class_type(t):
        if is_reference_type(t):
            return 'Ref[%s] ' % strip_name(t)
        else:
            return '%s *' % strip_name(t)
    else:
        if t == 'int':
            return 'int64_t '
        if t == 'float' or t == 'real':
            return 'real_t '
        return '%s ' % strip_name(t)


def generate_class_context(class_name, class_def, used_types, node_types, resource_types):
    includes = set()
    forwards = set()

    def detect_package(name):
        if name in node_types:
            return 'nodes'
        elif name in resource_types:
            return 'resources'
        return 'engine'

    for used_type in used_types:
        if is_enum(used_type) and is_nested_type(used_type):
            used_name = remove_enum_prefix(extract_nested_type(used_type))
            # imported_name = remove_nested_type_prefix(remove_enum_prefix(used_type))
            package = detect_package(used_name)

            if used_name != class_name:
                forwards.add((used_name, package))
        else:
            used_name = remove_enum_prefix(used_type)

            if used_name not in CORE_TYPES and used_name != class_name:
                package = detect_package(used_name)
                forwards.add((used_name, package))

    if class_def['base_class']:
        base_class = class_def['base_class']
        assert base_class not in CORE_TYPES
        forwards.add((base_class, detect_package(base_class)))

    prepared_methods = []

    for method in class_def['methods']:
        method_name = method.pop('name')
        method_name = escape_cpp(method_name)
        return_type = make_gdnative_type(method['return_type'])
        prepared_args = []

        sigs = []
        has_default_argument = False
        for arg in method['arguments']:
            arg_type = make_gdnative_type(arg['type'])
            arg_name = escape_cpp(arg['name'])
            arg_default = None

            if arg['has_default_value'] or has_default_argument:
                arg_default = escape_default_arg(arg['type'], arg['default_value'])
                has_default_argument = True

            if arg_default is not None:
                sig = 'const %s%s=%s' % (arg_type, arg_name, arg_default)
            else:
                sig = 'const %s%s' % (arg_type, arg_name)

            prepared_args.append((arg_name, arg_type, arg_default))
            sigs.append(sig)

        if method["has_varargs"]:
            sigs.append('...')

        prepared_methods.append((method_name, return_type, prepared_args, ', '.join(sigs)))

    return class_name, class_def, includes, forwards, prepared_methods


def escape_default_arg(_type, default_value):
    if _type == 'Color':
        return 'Color(%s)' % default_value
    elif _type in ('bool', 'int'):
        return default_value
    elif _type in ('Array', 'PoolVector2Array', 'PoolStringArray', 'PoolVector3Array', 'PoolColorArray',
                   'PoolIntArray', 'PoolRealArray', 'Transform', 'Transform2D', 'RID'):
        return '%s()' % _type
    elif _type in ('Vector2', 'Vector3', 'Rect2'):
        return '%s %s' % (_type, default_value)
    elif _type == 'Variant':
        return 'Variant()' if default_value == 'Null' else default_value
    elif _type == 'String':
        return '"%s"' % default_value

    elif default_value == 'Null' or default_value == '[Object:null]':
        return 'NULL'
    elif _type == 'Dictionary' and default_value == '{}':
        return 'NULL'

    return default_value


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
    for method in cls['methods']:
        if is_class_type(method['return_type']) and not (method['return_type'] in classes):
            classes.add(method['return_type'])

        for arg in method['arguments']:
            if is_class_type(arg['type']) and not (arg['type'] in classes):
                classes.add(arg['type'])
    return classes


def is_reference_type(name):
    return name in reference_types


def is_class_type(name):
    return name not in CORE_TYPES and name not in PRIMITIVE_TYPES


def is_core_type(name):
    return name in CORE_TYPES


def is_primitive(name):
    return name in PRIMITIVE_TYPES


def strip_name(name):
    return name.lstrip('_')


def extract_nested_type(nested_type):
    return nested_type[:nested_type.find('::')].lstrip('_')


def remove_nested_type_prefix(name):
    return name if '::' not in name else name[name.find('::') + 2:].lstrip('_')


def remove_enum_prefix(name):
    return re.sub(r'^enum\.', '', name).lstrip('_')


def is_nested_type(name, type=''):
    return ('%s::' % type) in name


def is_enum(name):
    return name.startswith('enum.')


def escape_cpp(name):
    if name in CYTHON_ESCAPES:
        return CYTHON_ESCAPES[name]

    return name


def python_module_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z]|2D|3D)', r'\1_\2', s1).lower()
    s3 = re.sub('_([23]d)', r'\1', s2)  # Don't separate _2d and _3d
    return re.sub('^gd_', 'gd', s3)  # Fix gd_ prefix
