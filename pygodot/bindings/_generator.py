import os
import re
import json
import struct
import hashlib
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
    'in':       'in_',
    'pass':     'pass_',
    'raise':    'raise_',
    'global':   'global_',
    'import':   'import_',
    'object':   'object_',
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

    # C++ bindings
    cpp_bindings_template = Template(filename=os.path.join(bindings_dir, 'templates', '_cpp_bindings.pxd.mako'))
    cpp_path = os.path.join(bindings_dir, '_cpp_bindings.pxd')
    cpp_source = cpp_bindings_template.render(classes=[generate_cppclass_context(c) for c in classes])
    with open(cpp_path, 'w', encoding='utf-8') as fp:
        fp.write(cpp_source)

    for module, _types in (('nodes', node_types), ('resources', resource_types), ('engine', engine_types)):
        write_cpp_definitions(module, _types, class_names)

    # Cython bindings
    write_cython_icall_definitions(generate_icalls_context(classes), root_dir)

    class_contexts = [generate_class_context(c) for c in classes]

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

    # Python bindings
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


def write_cython_icall_definitions(prepared_icalls, root_dir):
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

            key = '#'.join([strip_name(c['name']), escape_cython(method['name'])])
            icall2methodkeys.setdefault((ret, args), []).append(key)

    prepared_icalls = []

    for ret_type, args in icalls:
        methodkeys = icall2methodkeys[ret_type, args]
        icall_name = get_icall_name(ret_type, args)

        for key in methodkeys:
            icall_names[key] = icall_name

        sig = [get_icall_return_type(ret_type), icall_name, '(', 'godot_method_bind *mb, ', 'godot_object *o',
               *create_icall_arguments(args), ')']

        pxd_sig = [get_icall_pxd_return_type(ret_type), icall_name, '(', 'godot_method_bind*, ',
                   'godot_object*', *create_icall_arguments(args, is_pxd=True), ')']

        prepared_icalls.append((ret_type, args, ''.join(sig), ''.join(pxd_sig)))

    return prepared_icalls


def create_icall_arguments(args, is_pxd=False):
    sig = []

    for i, arg in enumerate(args):
        prefix = ' '
        sig.append(', const ')
        if is_core_type(arg):
            prefix = ''
            if is_pxd:
                sig.append(arg + '&')
            else:
                sig.append(arg + ' &')
        elif arg == 'int':
            sig.append('int64_t')
        elif arg == 'float':
            sig.append('double')
        elif is_primitive(arg):
            sig.append(arg)
        else:
            prefix = ''
            if is_pxd:
                sig.append('godot_object*')  # argument.owner
            else:
                sig.append('godot_object *')  # argument.owner

        if not is_pxd:
            sig.append(prefix + 'arg%s' % i)

    return sig


def get_icall_return_type(t):
    if is_class_type(t):
        return 'PyObject *'
    elif t == 'int':
        return 'int64_t '
    elif t == 'float' or t == 'real':
        return 'double '

    return t + ' '


def get_icall_pxd_return_type(t):
    if is_class_type(t):
        return 'object '
    elif t == 'int ':
        return 'int64_t'
    elif t == 'float' or t == 'real':
        return 'double '

    return t + ' '


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


def generate_class_context(class_def):
    class_name = strip_name(class_def['name'])
    includes, forwards = detect_used_classes(class_def)

    prepared_methods = []

    for method in class_def['methods']:
        method_name = method['name']
        method_name = escape_cython(method_name)
        return_type = make_cython_gdnative_type(method['return_type'], is_virtual=method['is_virtual'], is_return=True)

        args = []
        sigs = []
        pxd_sigs = []
        has_default_argument = False

        for arg in method['arguments']:
            has_default = arg['has_default_value'] or has_default_argument
            arg_type = make_cython_gdnative_type(arg['type'], has_default=has_default)
            arg_name = escape_cython(arg['name'])

            arg_default = None

            # if has_default:
            #     arg_default = escape_cython_default_arg(arg['type'], arg['default_value'])
            #     has_default_argument = True

            if arg_default is not None:
                pxd_sig = '%s%s=*' % (arg_type, arg_name)
                sig = '%s%s=%s' % (arg_type, arg_name, arg_default)
            else:
                pxd_sig = sig = '%s%s' % (arg_type, arg_name)

            args.append((arg_type, arg_name, arg))

            pxd_sigs.append(pxd_sig)
            sigs.append(sig)

        if method['has_varargs']:
            pxd_sigs.append('tuple __var_args')
            sigs.append('tuple __var_args')

        return_stmt = 'return '
        if is_enum(method['return_type']):
            return_stmt = 'return <%s>' % return_type.rstrip()
        elif method['return_type'] == 'void':
            return_stmt = ''

        prepared_methods.append((method_name, method, return_type, ', '.join(pxd_sigs), ', '.join(sigs),
                                 args, return_stmt))

    return class_name, class_def, includes, forwards, prepared_methods


# 58 character alphabet used
alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


# Not used
def make_sha1_suffix(value):
    hash = hashlib.sha1(value.encode('utf-8'))
    # Base58-encode first 4 bytes of SHA1 hash
    i, = struct.unpack('!L', hash.digest()[:4])
    string = b""
    while i:
        i, idx = divmod(i, 58)
        string = alphabet[idx:idx+1] + string
    return string.decode('utf-8')


def escape_cython_default_arg(_type, default_value):
    if _type == 'Color':
        return 'Color(%s)' % default_value
    elif _type in ('bool', 'int'):
        return default_value
    elif _type in ('Array', 'Dictionary', 'PoolVector2Array', 'PoolStringArray', 'PoolVector3Array', 'PoolColorArray',
                   'PoolIntArray', 'PoolRealArray', 'Transform', 'Transform2D', 'RID'):
        return '%s()' % _type
    elif _type in ('Vector2', 'Vector3', 'Rect2'):
        return '%s%s' % (_type, default_value)
    elif _type == 'Variant':
        if default_value == 'Null':
            return 'Variant()'
        else:
            return '<const Variant>%s' % default_value
    elif _type == 'String':
        return '<String><const char *>"%s"' % default_value
    elif default_value == 'Null' or default_value == '[Object:null]':
        if is_class_type(_type):
            return 'None'
        else:
            return 'NULL'

    return default_value


def cython_nonempty_comparison(_type):
    if is_class_type(_type):
        return 'is not None'
    if _type in ('bool', 'int'):
        return '!= 0'
    elif _type == 'String':
        return '!= <const String><const char *>""'

    return '!= NULL'


def make_cython_gdnative_type(t, is_virtual=False, is_return=False, has_default=False):
    prefix = '' if is_return or has_default else 'const '
    if is_enum(t):
        enum_name = remove_enum_prefix(t).replace('::', '')
        return '%s ' % enum_name
    elif is_class_type(t):
        return '%s ' % strip_name(t)

    if t == 'int':
        return prefix + 'int64_t '
    if t == 'float' or t == 'real':
        return prefix + 'real_t '

    if is_virtual:
        # Python runtime exceptions should not be forgotten!
        if t == 'void':
            return 'object '
        elif t == 'bool':
            return 'int '

    return prefix + '%s ' % strip_name(t)


def generate_cppclass_context(class_def):
    class_name = strip_name(class_def['name'])
    includes, forwards = detect_used_classes(class_def)

    prepared_methods = []

    for method in class_def['methods']:
        method_name = method['name']
        method_name = escape_cython(method_name)
        return_type = make_cpp_gdnative_type(method['return_type'])

        args = []
        sigs = []
        pxd_sigs = []
        has_default_argument = False

        for arg in method['arguments']:
            arg_type = make_cpp_gdnative_type(arg['type'])
            arg_name = escape_cython(arg['name'])

            arg_default = None

            if arg['has_default_value'] or has_default_argument:
                arg_default = escape_cpp_default_arg(arg['type'], arg['default_value'])
                has_default_argument = True

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


def make_cpp_gdnative_type(t):
    if is_enum(t):
        return 'int '  # '%s ' % remove_enum_prefix(t).replace('::', '.')
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


def escape_cpp_default_arg(_type, default_value):
    if _type == 'Color':
        return 'Color(%s)' % default_value
    elif _type in ('bool', 'int'):
        return default_value
    elif _type in ('Array', 'Dictionary', 'PoolVector2Array', 'PoolStringArray', 'PoolVector3Array', 'PoolColorArray',
                   'PoolIntArray', 'PoolRealArray', 'Transform', 'Transform2D', 'RID'):
        return '%s()' % _type
    elif _type in ('Vector2', 'Vector3', 'Rect2'):
        return '%s%s' % (_type, default_value)
    elif _type == 'Variant':
        return 'Variant()' if default_value == 'Null' else default_value
    elif _type == 'String':
        return '<String><const char *>"%s"' % default_value
    elif default_value == 'Null' or default_value == '[Object:null]':
        return 'NULL'

    return default_value


def detect_used_classes(class_def):
    class_name = strip_name(class_def['name'])
    used_types = _get_used_types(class_def)

    includes = set()
    forwards = set()

    for used_type in used_types:
        if is_enum(used_type) and is_nested_type(used_type):
            used_name = remove_enum_prefix(extract_nested_type(used_type))
            # imported_name = remove_nested_type_prefix(remove_enum_prefix(used_type))

            if used_name != class_name:
                forwards.add(used_name)
        else:
            used_name = remove_enum_prefix(used_type)

            if used_name not in CORE_TYPES and used_name != class_name:
                forwards.add(used_name)

    if class_def['base_class']:
        base_class = class_def['base_class']
        includes.add(base_class)

    return includes, forwards


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


def escape_cython(name):
    if name in CYTHON_ESCAPES:
        return CYTHON_ESCAPES[name]

    return name


def collect_dependent_types(forwards, includes, class_name, base_class):
    dependent_types = set(forwards) | set(includes)
    dependent_types.add(class_name)
    if base_class:
        dependent_types.add(base_class)

    return dependent_types


ARGUMENT_TYPE_FIXES = {
    'MenuButton': ('InputEvent', 'InputEventKey')
}


def clean_signature(signature, class_name):
    ret = signature

    # Inherited class methods can't change its arument types
    if class_name in ARGUMENT_TYPE_FIXES:
        orig, fixed = ARGUMENT_TYPE_FIXES[class_name]
        ret = ret.replace(orig, fixed)

    return ret


def python_module_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z]|2D|3D)', r'\1_\2', s1).lower()
    s3 = re.sub('_([23]d)', r'\1', s2)  # Don't separate _2d and _3d
    return re.sub('gd_([a-z]+)', r'gd\1', s3)  # Fix gd_ prefix