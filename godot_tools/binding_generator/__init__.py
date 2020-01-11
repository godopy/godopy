import os
import re
import sys
import json
from collections import defaultdict

from mako.template import Template


CYTHON_AUTOMATIC_CAST_TYPES = ('Variant', 'Array', 'Dictionary')


HAS_TYPE_CONVERTORS = (
    'Color', 'Vector2', 'Vector3', 'PoolByteArray', 'PoolIntArray',
    'PoolRealArray', 'PoolStringArray', 'PoolVector2Array',
    'PoolVector3Array', 'PoolColorArray'
)

NUMPY_CAST_TYPES = ('PoolStringArray',)

CORE_TYPES = (
    'Basis', 'Color', 'Dictionary', 'Error', 'NodePath', 'Plane',
    'Quat', 'Rect2', 'AABB', 'RID', 'String', 'Transform',
    'Transform2D', 'Variant', 'Vector2', 'Vector3',
    'Array', 'PoolByteArray', 'PoolIntArray', 'PoolRealArray', 'PoolStringArray', 'PoolVector2Array',
    'PoolVector3Array', 'PoolColorArray'
)

PRIMITIVE_TYPES = ('int', 'bool', 'real', 'float', 'void')

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
    'new':      '_new',
    'from':     '_from',
    'with':     '_with',
    'in':       '_in',
    'pass':     '_pass',
    'raise':    '_raise',
    'global':   '_global',
    'import':   '_import',
    'object':   '_object',
    'property': '_property',
    'get_singleton': '_get_singleton',  # Some API methods are in conflict with auto-generated get_singleton() methods
}

SPECIAL_ESCAPES = {
    'new': '__call__'
}

CYTHON_ESCAPES = {**CPP_ESCAPES, **CYTHON_ONLY_ESCAPES}

reference_types = set()
icall_names = {}

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
bindings_dir = os.path.join(root_dir, '_lib', 'godot', 'bindings')
templates_dir = os.path.join(root_dir, 'godot_tools', 'binding_generator', 'templates')


def write_api_pxd(echo=print):
    from .pxd_writer import PxdWriter, parse as parse_c_header

    output_dir = os.path.join(root_dir, '_lib', 'godot_headers')

    if not os.path.isdir(output_dir):
        echo(f'"{output_dir}" does not exist. Something went wrong…')
        sys.exit(1)

    echo("Converting 'gdnative_api_struct.gen.h' -> 'gdnative_api.pxd'")

    inpath = os.path.join(output_dir, 'gdnative_api_struct.gen.h')
    if not os.path.exists(inpath):
        echo(f'Required "gdnative_api_struct.gen.h" file doesn\'t exist in "{output_dir}"')
        sys.exit(1)

    initial_dir = os.getcwd()

    os.chdir(output_dir)
    fname = 'gdnative_api_struct.gen.h'

    with open(fname, 'r') as infile:
        code = infile.read()

    extra_cpp_args = ['-I', '.']
    if sys.platform == 'darwin':
        # TODO: Use 'xcode-select -p' output
        extra_cpp_args += ['-I', "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include"]

    p = PxdWriter(fname)
    p.visit(parse_c_header(code, extra_cpp_args=extra_cpp_args))

    pxd = 'from libc.stdint cimport {:s}\n'.format(', '.join(p.stdint_declarations))
    pxd += 'from libc.stddef cimport wchar_t\nfrom libcpp cimport bool\n\n'
    pxd += str(p)
    pxd = pxd.replace('uint8_t _dont_touch_that[]', 'pass')
    pxd = pxd.replace('extern from "gdnative_api_struct.gen.h":', 'extern from "gdnative_api_struct.gen.h" nogil:')

    with open('gdnative_api.pxd', 'w', encoding='utf-8') as f:
        f.write(pxd)

    with open('__init__.py', 'w', encoding='utf-8') as f:
        pass

    os.chdir(initial_dir)


def generate(generate_cpp=True, generate_cython=True, generate_python=True, echo=print, preloaded_classes=None):
    if preloaded_classes:
        classes = preloaded_classes

    with open(os.path.join(root_dir, '_lib', 'godot_headers', 'api.json'), encoding='utf-8') as fp:
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

    singleton_types = set(strip_name(c['name']) for c in classes if c['singleton'])
    tools_types = set(strip_name(c['name']) for c in classes if c['api_type'] == 'tools') - singleton_types
    node_types = set(['Node', 'TreeItem', *clsdict['Node']['child_classes']]) - tools_types - singleton_types
    resource_types = set(['Resource', 'TriangleMesh', *clsdict['Resource']['child_classes']]) - tools_types - singleton_types
    core_api_types = set(strip_name(c['name']) for c in classes) - tools_types - node_types - resource_types - singleton_types

    if generate_cython or generate_python:
        if generate_cython and generate_python:
            bindings_name = 'Python and Cython'
        elif generate_cython:
            bindings_name = 'Cython'
        else:
            bindings_name = 'Python'

        echo("Generating %s bindings…" % bindings_name)
        echo('Singleton classes: %d' % len(singleton_types))
        echo('Node classes: %d' % len(node_types))
        echo('Resource classes: %d' % len(resource_types))
        echo('Core API classes: %d' % len(core_api_types))
        echo('Tools classes: %d' % len(tools_types))

    module_data = (
        ('singletons', singleton_types),
        ('nodes', node_types),
        ('resources', resource_types),
        ('core', core_api_types),
        ('tools', tools_types)
    )

    if generate_cpp:
        # C++ bindings
        class_contexts = [generate_class_context(c, 'cpp') for c in classes]
        cpp_bindings_template = Template(filename=os.path.join(templates_dir, '_cpp_bindings.pxd.mako'))
        cpp_path = os.path.join(bindings_dir, '_cpp_bindings.pxd')
        cpp_source = cpp_bindings_template.render(classes=class_contexts)
        with open(cpp_path, 'w', encoding='utf-8') as fp:
            fp.write(cpp_source)

        for module, _types in module_data:
            write_cpp_definitions(module, _types, class_names, class_contexts)

    if generate_cython:
        class_contexts = [generate_class_context(c, 'cython') for c in classes]

        # Cython bindings
        write_icall_definitions(generate_icalls_context(classes, 'cython'), 'cython', root_dir)

        cython_bindings_pxd = (
            os.path.join(bindings_dir, '_cython_bindings.pxd'),
            Template(filename=os.path.join(templates_dir, '_cython_bindings.pxd.mako'))
        )

        cython_bindings_pyx = (
            os.path.join(bindings_dir, '_cython_bindings.pyx'),
            Template(filename=os.path.join(templates_dir, '_cython_bindings.pyx.mako'))
        )

        for path, template in (cython_bindings_pxd, cython_bindings_pyx):
            source = template.render(classes=class_contexts, icall_names=icall_names)
            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(source)

        for module, _types in module_data:
            write_cython_definitions(module, _types, class_names, class_contexts)

    if generate_python:
        class_contexts = [generate_class_context(c, 'python') for c in classes]

        # Python bindings
        write_icall_definitions(generate_icalls_context(classes, 'python'), 'python', root_dir)

        python_bindings_pxd = (
            os.path.join(bindings_dir, '_python_bindings.pxd'),
            Template(filename=os.path.join(templates_dir, '_python_bindings.pxd.mako'))
        )

        python_bindings_pyx = (
            os.path.join(bindings_dir, '_python_bindings.pyx'),
            Template(filename=os.path.join(templates_dir, '_python_bindings.pyx.mako'))
        )

        for path, template in (python_bindings_pxd, python_bindings_pyx):
            source = template.render(classes=class_contexts, icall_names=icall_names)
            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(source)
        for module, _types in module_data:
            write_python_definitions(module, _types, class_names, class_contexts)


def write_python_definitions(python_module, _types, class_names, class_contexts):
    python_package_template = Template(filename=os.path.join(templates_dir, 'python_module.py.mako'))

    python_path = os.path.join(bindings_dir, 'python', '%s.py' % python_module)
    classes = {ctx[0]: ctx[1] for ctx in class_contexts if ctx[0] in _types}
    class_names = [cn for cn in class_names if cn in _types]
    python_source = python_package_template.render(class_names=class_names, classes=classes)

    with open(python_path, 'w', encoding='utf-8') as fp:
        fp.write(python_source)


def write_cython_definitions(cython_module, _types, class_names, class_contexts):
    cython_package_template = Template(filename=os.path.join(templates_dir, 'cython_module.pxd.mako'))

    cython_path = os.path.join(bindings_dir, 'cython', '%s.pxd' % cython_module)
    classes = {ctx[0]: ctx[1] for ctx in class_contexts if ctx[0] in _types}
    class_names = [cn for cn in class_names if cn in _types]
    cython_source = cython_package_template.render(class_names=class_names, classes=classes)

    with open(cython_path, 'w', encoding='utf-8') as fp:
        fp.write(cython_source)


def write_cpp_definitions(cpp_module, _types, class_names, class_contexts):
    cpp_package_template = Template(filename=os.path.join(templates_dir, 'cpp_module.pxd.mako'))

    cpp_path = os.path.join(bindings_dir, 'cpp', '%s.pxd' % cpp_module)
    classes = {ctx[0]: ctx[1] for ctx in class_contexts if ctx[0] in _types}
    class_names = [cn for cn in class_names if cn in _types]
    cpp_source = cpp_package_template.render(class_names=class_names, classes=classes)

    with open(cpp_path, 'w', encoding='utf-8') as fp:
        fp.write(cpp_source)


def write_icall_definitions(prepared_icalls, binding, root_dir):
    icalls_header_path = os.path.join(root_dir, 'include', 'pygen', '__%s_icalls.hpp' % binding)
    icalls_pxd_path = os.path.join(bindings_dir, binding, '__icalls.pxd')

    icalls_header_template = Template(filename=os.path.join(templates_dir, '__%s_icalls.hpp.mako' % binding))
    icalls_header = icalls_header_template.render(icalls=prepared_icalls)

    with open(icalls_header_path, 'w', encoding='utf-8') as fp:
        fp.write(icalls_header)

    icalls_pxd_template = Template(filename=os.path.join(templates_dir, '__%s_icalls.pxd.mako' % binding))
    icalls_pxd = icalls_pxd_template.render(icalls=prepared_icalls)

    with open(icalls_pxd_path, 'w', encoding='utf-8') as fp:
        fp.write(icalls_pxd)


def generate_icalls_context(classes, language):
    icalls = set()

    icall2methodkeys = {}

    for c in classes:
        for method in c['methods']:
            args = tuple(get_icall_type_name(arg['type']) for arg in method['arguments'])
            var_arg = None
            if method['has_varargs']:
                var_arg = '__var_args'

            ret = get_icall_type_name(method['return_type'])
            icalls.add((ret, args, var_arg))

            method_name = escape_python(method['name'])
            # vararg methods have two version, `cdef` version is '_'-prefixed
            key = '#'.join([strip_name(c['name']), method_name])
            icall2methodkeys.setdefault((ret, args, var_arg), []).append(key)

            if method['name'] in SPECIAL_ESCAPES:
                key_special = '#'.join([strip_name(c['name']), SPECIAL_ESCAPES[method['name']]])
                icall2methodkeys[ret, args, var_arg].append(key_special)
            elif method['has_varargs']:
                key_vararg = '#'.join([strip_name(c['name']), method_name])
                icall2methodkeys[ret, args, var_arg].append(key_vararg)

    prepared_icalls = []

    for ret_type, args, var_arg in icalls:
        methodkeys = icall2methodkeys[ret_type, args, var_arg]
        icall_name = get_icall_name(language, ret_type, args, var_arg)

        for key in methodkeys:
            icall_names[key] = icall_name

        sig = [
            get_icall_return_type(ret_type, has_varargs=var_arg), icall_name,
            '(', 'godot_method_bind *mb, ', 'godot_object *o',
            *create_icall_arguments(args, var_arg), ')'
        ]

        pxd_sig = [
            get_icall_pxd_return_type(ret_type, has_varargs=var_arg), icall_name,
            '(', 'godot_method_bind*, ', 'godot_object*',
            *create_icall_arguments(args, var_arg, is_pxd=True), ')'
        ]

        prepared_icalls.append((ret_type, args, var_arg, ''.join(sig), ''.join(pxd_sig)))

    return prepared_icalls


def create_icall_arguments(args, var_arg, is_pxd=False):
    sig = []

    for i, arg in enumerate(args):
        prefix = ' '
        sig.append(', const ')
        if is_core_type(arg):
            prefix = ''
            if is_pxd:
                sig.append('%s&' % arg)
            else:
                sig.append('%s &' % arg)
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

    if var_arg:
        if is_pxd:
            sig.append(', const Array')
        else:
            sig.append(', const Array __var_args')

    return sig


def get_icall_return_type(t, has_varargs=False):
    if is_class_type(t):
        return 'Variant ' if has_varargs else 'PyObject *'
    elif t == 'int':
        return 'int64_t '
    elif t == 'float' or t == 'real':
        return 'double '

    return t + ' '


def get_icall_pxd_return_type(t, has_varargs=False):
    if is_class_type(t):
        return 'Variant ' if has_varargs else 'object '
    elif t == 'int ':
        return 'int64_t'
    elif t == 'float' or t == 'real':
        return 'double '

    return t + ' '


def get_icall_name(lang, ret_type, args, var_arg):
    name = "___%s_icall_" % lang
    name += strip_name(ret_type)
    for arg in args:
        name += "_" + strip_name(arg)

    if var_arg:
        name += var_arg

    return name


def get_icall_type_name(name):
    if name.startswith('enum'):
        return 'int'
    if is_class_type(name):
        return 'Object'
    return name


def generate_class_context(class_def, language):
    if language == 'cpp':
        return generate_cppclass_context(class_def)

    class_name = strip_name(class_def['name'])
    includes, forwards = detect_used_classes(class_def)

    prepared_methods = []

    for method in class_def['methods']:
        method_name = method['name']
        method_name = escape_python(method_name)
        return_type = make_cython_gdnative_type(method['return_type'], is_virtual=method['is_virtual'], is_return=True)

        args = []
        sigs = []
        def_sigs = []
        pxd_sigs = []
        init_args = []
        has_default_argument = False

        for arg in method['arguments']:
            has_default = arg['has_default_value'] or has_default_argument
            if has_default_argument and not arg['has_default_value']:
                print('Update "has_default_value" flag for %s of %s', (arg['name'], method_name))
                arg['has_default_value'] = True
            if language == 'cython':
                arg_type = make_cython_gdnative_type(arg['type'], has_default=has_default)
            else:
                arg_type = make_python_gdnative_type(arg['type'], has_default=has_default)
            arg_name = escape_python(arg['name'])

            arg_default = None
            arg_init = None
            if has_default:
                arg_default = escape_cython_default_arg(arg['type'], arg['default_value'])
                has_default_argument = True
                if arg_init:
                    real_arg_name = arg_name
                    arg_name = '_' + real_arg_name
                    init_args.append((arg_type, real_arg_name, arg_name, arg_init))

            if arg_default is not None:
                pxd_sig = '%s%s=*' % (arg_type, arg_name)
                sig = '%s%s=%s' % (arg_type, arg_name, arg_default)
            else:
                pxd_sig = sig = '%s%s' % (arg_type, arg_name)

            args.append((arg_type, arg_name, arg, arg_init))

            pxd_sigs.append(pxd_sig)
            sigs.append(sig)
            def_sigs.append(sig)

        if method['has_varargs']:
            pxd_sigs.append('cpp.Array __var_args')
            sigs.append('cpp.Array __var_args')
            def_sigs.append('*__var_args')

        return_stmt = 'return '
        if language == 'cython' and is_enum(method['return_type']):
            return_stmt = 'return <%s>' % return_type.rstrip()
        # elif language == 'python' and not is_enum(method['return_type']) and is_class_type(method['return_type']):
        #     return_stmt = 'return <%s>' % return_type.rstrip()
        elif method['return_type'] == 'void':
            return_stmt = ''

        defmethod = {k: v for k, v in method.items()}
        method['__func_type'] = 'cdef'
        defmethod['__func_type'] = 'def'
        if language == 'cython':
            prepared_methods.append((
                method_name, method, return_type,
                ', '.join(pxd_sigs), ', '.join(sigs),
                args, return_stmt, init_args
            ))
        elif method['name'] not in SPECIAL_ESCAPES:
            prepared_methods.append((
                method_name, defmethod, return_type,
                '<no-pxd>', ', '.join(def_sigs),
                args, return_stmt, init_args
            ))

        if method['name'] in SPECIAL_ESCAPES:
            return_stmt += '<object>'
            prepared_methods.append((
                SPECIAL_ESCAPES[method['name']], defmethod, return_type,
                '<no-pxd>', ', '.join(def_sigs),
                args, return_stmt, init_args
            ))

    return class_name, class_def, includes, forwards, prepared_methods


def escape_cython_default_arg(_type, default_value):
    if _type == 'Color':
        return 'py.Color(%s)' % default_value
    elif _type in ('bool', 'int'):
        return default_value
    elif _type in ('Array', 'Dictionary', 'PoolVector2Array', 'PoolStringArray', 'PoolVector3Array', 'PoolColorArray',
                   'PoolIntArray', 'PoolRealArray', 'Transform', 'Transform2D', 'RID'):
        return 'py.%s()' % _type
    elif _type in ('Vector2', 'Vector3', 'Rect2'):
        return 'py.%s%s' % (_type, default_value)
    elif _type == 'Variant':
        if default_value == 'Null':
            return 'None'
        else:
            return '(<object>%s)' % default_value
    elif _type == 'String':
        # return 'NULL', 'String(%r)' % default_value
        return repr(default_value)
    elif default_value == 'Null' or default_value == '[Object:null]':
        return 'None'

    return default_value


def make_cython_gdnative_type(t, is_virtual=False, is_return=False, has_default=False):
    prefix = '' if is_return or has_default else 'const '
    if is_enum(t):
        enum_name = remove_enum_prefix(t).replace('::', '')
        return '%s ' % enum_name
    elif is_class_type(t):
        return '%s ' % strip_name(t)
    elif t == 'String':
        return 'str '
    elif has_default and t in CYTHON_AUTOMATIC_CAST_TYPES:
        return 'object '
    elif has_default and is_core_type(t):
        return 'py.%s ' % strip_name(t)

    elif t == 'int':
        return prefix + 'int '
    elif t == 'float' or t == 'real':
        return prefix + 'float '

    elif t == 'void':
        return 'object '
    elif t == 'bool':
        return 'bint '
    else:
        return prefix + 'cpp.%s ' % strip_name(t)


def make_python_gdnative_type(t, is_virtual=False, is_return=False, has_default=False):
    if is_enum(t):
        return 'int '
    elif is_class_type(t):
        # return 'object '
        return '%s ' % strip_name(t)
    elif t == 'String':
        return 'str '
    elif t == 'Dictionary':
        return 'object '
    elif t == 'Array':
        return 'object '
    elif t == 'Variant':
        return 'object '
    elif t in HAS_TYPE_CONVERTORS:
        return 'object '
    elif is_core_type(t):
        return 'py.%s ' % strip_name(t)

    elif t == 'int':
        return 'int '
    elif t == 'float' or t == 'real':
        return 'float '

    elif t == 'void':
        return 'object '
    elif t == 'bool':
        return 'bint '
    else:
        raise ValueError('Uknown Type %r' % t)


def generate_cppclass_context(class_def):
    class_name = strip_name(class_def['name'])
    includes, forwards = detect_used_classes(class_def)

    prepared_methods = []

    for method in class_def['methods']:
        method_name = method['name']
        method_name = escape_cpp(method_name)
        return_type = make_cpp_gdnative_type(method['return_type'])

        args = []
        sigs = []
        pxd_sigs = []
        # has_default_argument = False

        for arg in method['arguments']:
            arg_type = make_cpp_gdnative_type(arg['type'])
            arg_name = escape_cpp(arg['name'])

            arg_default = None

            # FIXME: Cython does not work correctly with default C++ arguments yet
            # if arg['has_default_value'] or has_default_argument:
            #     arg_default = escape_cpp_default_arg(arg['type'], arg['default_value'])
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
        return '<String>"%s"' % default_value
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


def escape_cpp(name):
    if name in CYTHON_ESCAPES:
        return CYTHON_ESCAPES[name]

    return name


def escape_python(name):
    name = name.replace('/', '__')
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
