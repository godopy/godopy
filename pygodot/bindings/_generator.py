import os
import re
import json
from collections import defaultdict

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
    'with':     'with_',
    'in':     'in_',
    'pass':     'pass_',
    'raise':    'raise_',
    'global':   'global_',
    'import':   'import_',
    'get_singleton': '_get_singleton',  # Some API methods are in conflict with auto-generated get_singleton() methods
    'set_singleton': '_set_singleton',  # for consistency
}

CYTHON_ESCAPES = {**CPP_ESCAPES, **CYTHON_ONLY_ESCAPES}

reference_types = set()
icall_names = {}

bindings_dir = os.path.abspath(os.path.dirname(__file__))


def main(root_dir, echo=print):
    with open(os.path.join(root_dir, 'godot_headers', 'api.json'), encoding='utf-8') as fp:
        classes = json.load(fp)

    class_names = [strip_name(c['name']) for c in classes]

    for c in classes:
        if c['is_reference']:
            reference_types.add(strip_name(c['name']))

    # map of [classname] => set of all child classes
    children = defaultdict(set)

    clsdict = {strip_name(c['name']): c for c in classes}

    def set_children(parentname, cname):
        if parentname:
            children[parentname].add(cname)
            set_children(clsdict[parentname]['base_class'], cname)
            set_children(clsdict[parentname]['base_class'], parentname)

    for c in classes:
        cname = strip_name(c['name'])
        parentname = strip_name(c['base_class'])
        set_children(parentname, cname)

    for c in classes:
        child_classes = tuple(children[strip_name(c['name'])])
        c['weight'] = len(child_classes)
        c['child_classes'] = child_classes

    classes.sort(key=lambda c: c['weight'], reverse=True)

    node_types = set(['Node', 'TreeItem', *clsdict['Node']['child_classes']])
    resource_types = set(['Resource', 'TriangleMesh', *clsdict['Resource']['child_classes']])
    engine_types = set([strip_name(c['name']) for c in classes]) - node_types - resource_types

    print('Node classes:', len(node_types))
    print('Resource classes:', len(resource_types))
    print('Engine classes:', len(engine_types))

    class_contexts = []

    for class_def in classes:
        used_types = _get_used_types(class_def)
        class_name = strip_name(class_def['name'])
        class_contexts.append(generate_class_context(class_name, class_def, used_types, node_types, resource_types))

    write_icall_definitions(generate_icalls_context(classes), root_dir)

    cpp_bindings_template = Template(filename=os.path.join(bindings_dir, 'templates', '_cpp_bindings.pxd.mako'))
    cpp_path = os.path.join(bindings_dir, '_cpp_bindings.pxd')

    cpp_source = cpp_bindings_template.render(classes=class_contexts)
    with open(cpp_path, 'w', encoding='utf-8') as fp:
        fp.write(cpp_source)

    for module, _types in (('nodes', node_types), ('resources', resource_types), ('engine', engine_types)):
        write_cpp_definitions(module, _types, class_names)

    cython_bindings_pxd = (
        os.path.join(bindings_dir, '_cython_bindings.pxd'),
        Template(filename=os.path.join(bindings_dir, 'templates', '_cython_bindings.pxd.mako'))
    )

    cython_bindings_pyx = (
        os.path.join(bindings_dir, '_cython_bindings.pyx'),
        Template(filename=os.path.join(bindings_dir, 'templates', '_cython_bindings.pyx.mako'))
    )

    for path, template in (cython_bindings_pxd, cython_bindings_pyx):
        source = template.render(classes=class_contexts, icall_names=icall_names)
        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(source)

    for module, _types in (('nodes', node_types), ('resources', resource_types), ('engine', engine_types)):
        write_cython_definitions(module, _types, class_names)

    python_bindings_template = Template(filename=os.path.join(bindings_dir, 'templates', '_python_bindings.pyx.mako'))
    python_path = os.path.join(bindings_dir, '_python_bindings.pyx')

    python_source = python_bindings_template.render(classes=class_contexts)
    with open(python_path, 'w', encoding='utf-8') as fp:
        fp.write(python_source)

    for module, _types in (('nodes', node_types), ('resources', resource_types), ('engine', engine_types)):
        write_python_definitions(module, _types, class_names)


def write_python_definitions(python_module, _types, class_names):
    python_package_template = Template(filename=os.path.join(bindings_dir, 'templates', 'python_module.py.mako'))

    python_path = os.path.join(bindings_dir, '%s.py' % python_module)
    python_source = python_package_template.render(class_names=[cn for cn in class_names if cn in _types])

    with open(python_path, 'w', encoding='utf-8') as fp:
        fp.write(python_source)


def write_cython_definitions(cython_module, _types, class_names):
    cython_package_template = Template(filename=os.path.join(bindings_dir, 'templates', 'cython_module.pxd.mako'))

    cpp_path = os.path.join(bindings_dir, 'cython', '%s.pxd' % cython_module)
    cpp_source = cython_package_template.render(class_names=[cn for cn in class_names if cn in _types])

    with open(cpp_path, 'w', encoding='utf-8') as fp:
        fp.write(cpp_source)


def write_cpp_definitions(cpp_module, _types, class_names):
    cpp_package_template = Template(filename=os.path.join(bindings_dir, 'templates', 'cpp_module.pxd.mako'))

    cpp_path = os.path.join(bindings_dir, 'cpp', '%s.pxd' % cpp_module)
    cpp_source = cpp_package_template.render(class_names=[cn for cn in class_names if cn in _types])

    with open(cpp_path, 'w', encoding='utf-8') as fp:
        fp.write(cpp_source)


def write_icall_definitions(prepared_icalls, root_dir):
    icalls_header_path = os.path.join(root_dir, 'include', 'pygen', '__cython_icalls.hpp')
    icalls_pxd_path = os.path.join(bindings_dir, 'cython', '__icalls.pxd')

    icalls_header_template = Template(filename=os.path.join(bindings_dir, 'templates', '__cython_icalls.hpp.mako'))
    icalls_header = icalls_header_template.render(icalls=prepared_icalls)

    with open(icalls_header_path, 'w', encoding='utf-8') as fp:
        fp.write(icalls_header)

    icalls_pxd_template = Template(filename=os.path.join(bindings_dir, 'templates', '__cython_icalls.pxd.mako'))
    icalls_pxd = icalls_pxd_template.render(icalls=prepared_icalls)

    with open(icalls_pxd_path, 'w', encoding='utf-8') as fp:
        fp.write(icalls_pxd)


def generate_icalls_context(classes):
    icalls = set()

    icall2methodkeys = {}

    for c in classes:
        for method in c['methods']:
            if method['has_varargs']:
                continue
            args = tuple(get_icall_type_name(arg['type']) for arg in method['arguments'])
            ret = get_icall_type_name(method['return_type'])
            icalls.add((ret, args))

            key = '#'.join([strip_name(c['name']), escape_cpp(method['name'])])
            icall2methodkeys.setdefault((ret, args), []).append(key)

    prepared_icalls = []

    for ret_type, args in icalls:
        methodkeys = icall2methodkeys[ret_type, args]
        for key in methodkeys:
            icall_names[key] = get_icall_name(ret_type, args)

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

        signature = ''.join(sig)
        prepared_icalls.append((ret_type, args, signature, ret_type != 'void'))

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
        includes.add((base_class, detect_package(base_class)))

    prepared_methods = []

    for method in class_def['methods']:
        method_name = method['name']
        method_name = escape_cpp(method_name)
        return_type = make_gdnative_type(method['return_type'])

        args = []
        sigs = []
        pxd_sigs = []
        has_default_argument = False

        for arg in method['arguments']:
            arg_type = make_gdnative_type(arg['type'])
            arg_name = escape_cpp(arg['name'])

            arg_default = None

            # TODO: reenable default args, requires special handling for differnet types
            # Cython would pass NULL for default String args in C++ class methods, skip them
            # if (arg['has_default_value'] and arg['type'] != 'String') or has_default_argument:
            #     arg_default = escape_default_arg(arg['type'], arg['default_value'])
            #     has_default_argument = True

            if arg_default is not None:
                pxd_sig = 'const %s%s=*' % (arg_type, arg_name)
                sig = 'const %s%s=%s' % (arg_type, arg_name, arg_default)
            else:
                pxd_sig = sig = 'const %s%s' % (arg_type, arg_name)

            args.append((arg_type, arg_name))

            pxd_sigs.append(pxd_sig)
            sigs.append(sig)

        if method["has_varargs"]:
            pxd_sigs.append('...')
            sigs.append('...')

        prepared_methods.append((method_name, return_type, ', '.join(pxd_sigs), ', '.join(sigs), args))

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


def collect_dependent_types(forwards, includes, class_name, base_class):
    dependent_types = set(decl for decl, _ in forwards) | set(decl for decl, _ in includes)
    dependent_types.add(class_name)
    if base_class:
        dependent_types.add(base_class)

    return dependent_types


ARGUMENT_TYPE_FIXES = {
    'MenuButton': ('InputEvent', 'InputEventKey')
}


def clean_signature(signature, forwards, includes, class_name, base_class):
    dependent_types = collect_dependent_types(forwards, includes, class_name, base_class)
    ret = signature

    for cname in dependent_types:
        # Pass arguments as Python objects

        # `const Node *from_node=NULL` -> `Node from_node=None`
        ret = re.sub(r'const %s \*(\w+)\=NULL' % cname, r'%s \1=None' % cname, ret)

        ret = re.sub(r'const %s \*' % cname, '%s ' % cname, ret)
        ret = re.sub(r'const %s \*' % cname, '%s ' % cname, ret)

        # `const Ref[Texture] texture=NULL` -> `Texture texture=None`
        ret = re.sub(r'const Ref\[%s\] (\w+)\=NULL' % cname, r'%s \1=None' % cname, ret)
        ret = re.sub(r'const Ref\[%s\] ' % cname, '%s ' % cname, ret)

        ret = re.sub(r'const String (\w+)=\"(.*)\"', r'const String \1=<const String><const char *>"\2"', ret)
        ret = re.sub(r'const ([A-Z]\w+) (\w+)=NULL', r'const \1 \2=<const \1>NULL', ret)
        ret = re.sub(r'const ([A-Z]\w+) (\w+)=0', r'const \1 \2=<const \1>0', ret)

    # Inherited class methods can't change its arument types
    if class_name in ARGUMENT_TYPE_FIXES:
        orig, fixed = ARGUMENT_TYPE_FIXES[class_name]
        ret = ret.replace(orig, fixed)

    return ret


def clean_return_type(name, forwards, includes, class_name, base_class):
    dependent_types = collect_dependent_types(forwards, includes, class_name, base_class)
    ret = name.replace('.', '')  # enums
    ret = re.sub(r'Ref\[(.+)\]', r'\1', ret)
    for cname in dependent_types:
        # Return a Python object
        ret = re.sub(r'%s \*' % cname, '%s ' % cname, ret)
    return ret


def python_module_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z]|2D|3D)', r'\1_\2', s1).lower()
    s3 = re.sub('_([23]d)', r'\1', s2)  # Don't separate _2d and _3d
    return re.sub('gd_([a-z]+)', r'gd\1', s3)  # Fix gd_ prefix
