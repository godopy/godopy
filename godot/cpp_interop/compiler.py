import os
import sys
import importlib

from redbaron import RedBaron
from mako.template import Template
from Cython.Compiler.Main import CompilationOptions, default_options, compile as cython_compile

CPPDEFS_MODULE = 'godot_cpp'

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
# api_path = os.path.join(base_dir, 'godot_headers', 'api.json')
# with open(api_path, 'r') as f:
#     gdapi = {i['name']: i for i in json.load(f)}

templates_dir = os.path.join(base_dir, 'godot_cpp', 'templates')

def is_godot_cpp_import(node):
    return node.type == 'import' and joinval(node.value[0]) == CPPDEFS_MODULE

def is_register_class(node):
    return len(node.value) == 3 and str(node.value[1]) == 'register_class'

def is_decorated_function(node):
    return node.type == 'def' and node.decorators

GLOBAL_HINTS = 'cimport', 'register_class'
CLASS_HINTS = 'attr', 'method', 'register_property', 'register_method', 'register_signal'

COMMENT_HINTS = False
ENDL = RedBaron('\n\n')[0]

def compile(code, output_dir, name):
    tree = RedBaron(code)
    macro_prefix = ''
    basename = name
    classnames = set()

    def is_hint(node):
        if len(node.value) != 3:
            return False

        instance = str(node.value[0])
        method = str(node.value[1])
        if method == '.':
            method = str(node.value[2])

        if instance == macro_prefix and method in GLOBAL_HINTS:
            return True

        return instance in classnames and method in CLASS_HINTS

    def is_hinted_statement(node):
        return node.type in ('atomtrailers', 'assignment') and is_hint(node)

    for_deletion = []

    for i, node in enumerate(tree):
        if is_godot_cpp_import(node):
            macro_prefix = node.value[0].target or joinval(node.value[0])
        elif is_hint(node):
            if is_register_class(node):
                classnames.add(node.target.value)

            # Remove godot_cpp hints
            for_deletion.append(i)
        elif is_decorated_function(node):
            for j, dec in enumerate(node.decorators):
                if (is_hint(dec.value)):
                    # Remove godot_cpp decorator hints
                    if COMMENT_HINTS:
                        tree[i].decorators[j].type = 'atomtrailers'
                        tree[i].decorators[j].value = empty(tree[i].decorators[j].value)
                    else:
                        del tree[i].decorators[j]

    for i in reversed(for_deletion):
        if COMMENT_HINTS:
            tree[i] = empty(tree[i])
        else:
            tree[i] = ENDL

    sys.path.insert(0, output_dir)
    mod = importlib.import_module(basename)
    del sys.path[0]
    macro_mod = getattr(mod, macro_prefix)

    context = dict(basename=basename, registry=macro_mod.registry)

    pxd = Template(filename=os.path.join(templates_dir, 'impl.pxd.mako')).render(**context)
    hpp = Template(filename=os.path.join(templates_dir, 'wrap.hpp.mako')).render(**context)

    with open(os.path.join(output_dir, f'{basename}__impl.py'), 'w') as f:
        f.write(tree.dumps())
    with open(os.path.join(output_dir, f'{basename}__impl.pxd'), 'w') as f:
        f.write(pxd)
    with open(os.path.join(output_dir, f'{basename}__wrap.hpp'), 'w') as f:
        f.write(hpp)

    # Compile .py/.pxd, required for the next step
    cython_options = {}
    cython_options.update(default_options)
    cython_options['cplus'] = 1
    cython_options['output_file'] = os.path.join(output_dir, f'{basename}__impl.cpp')
    cython_options['language_level'] = 3
    cython_compile([os.path.join(output_dir, f'{basename}__impl.py')], CompilationOptions(cython_options))

    context['method_map'] = map_compiled_methods(macro_mod.registry.classes, output_dir, basename)

    wrapper_code = Template(filename=os.path.join(templates_dir, 'wrap.cpp.mako')).render(**context)
    with open(os.path.join(output_dir, f'{basename}__wrap.cpp'), 'w') as f:
        f.write(wrapper_code)


def map_compiled_methods(classes, output_dir, basename):
    method_map = {}
    for cls in classes:
        for method in cls.methods:
            method_map[method] = None

    header_chunks = []
    with open(os.path.join(output_dir, f'{basename}__impl.h'), 'r') as f:
        for line in f:
            if f'{basename}__impl' in line and '__PYX_EXTERN_C' in line:
                for chunk in line.split():
                    if f'{basename}__impl' in chunk:
                        header_chunks.append(chunk.split('(')[0])

    for method in method_map:
        for chunk in header_chunks:
            if method in chunk:
                method_map[method] = chunk
                continue

    return method_map

def joinval(node):
    return '.'.join(str(v) for v in node.value)

def empty(node):
    return RedBaron('# %s' % node)[0]

def argtype(arg):
    if not arg.annotation:
        return None
    typedesc = arg.annotation.value
    return typedesc[len(typedesc) - 1]

def returntype(arg):
    if not arg.return_annotation:
        return 'void'
    typedesc = arg.return_annotation.value
    return typedesc[len(typedesc) - 1].value
