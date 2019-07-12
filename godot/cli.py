import os
import sys
import json
import click

from . import get_version

from redbaron import RedBaron
from mako.template import Template
from Cython.Compiler.Main import CompilationOptions, default_options, compile as cython_compile

from collections import OrderedDict as odict

CPPDEFS_MODULE = 'godot_cpp'

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
api_path = os.path.join(base_dir, 'godot_headers', 'api.json')
with open(api_path, 'r') as f:
    gdapi = {i['name']: i for i in json.load(f)}

templates_dir = os.path.join(base_dir, 'godot_cpp', 'templates')

@click.group()
def pygodot():
    pass

def compile(tree, output_dir, name):
    macro_prefix = ''
    directives = set()
    cimports = odict()
    registered_classes = odict()
    basename = name

    for i, node in enumerate(tree):
        if node.type == 'import' and joinval(node.value[0]) == CPPDEFS_MODULE:
            macro_prefix = node.value[0].target or joinval(node.value[0])
            directives.add(macro_prefix)

        elif node.type in ('atomtrailers', 'assignment'):
            targets = None
            if node.type == 'assignment':
                if  node.target.type == 'tuple':
                    targets = node.target.value[0]
                else:
                    targets = (node.target,)
            if len(node.value) != 3 or str(node.value[0]) not in directives:
                continue

            name, method, args = node.value

            name = str(name)
            method = str(method)

            if method == 'register_class':
                name = str(targets[0])
                base = args[1].value.to_python()

                directives.add(name)
                assert base in gdapi, f'"{base}" class not found in Godot API'

                registered_classes[name] = {
                    'name': args[0].value.to_python(),
                    'base': base
                }

                cimports.setdefault('godot_cpp.gen', set()).add(base)
            elif method == 'cimport':
                module, symbols = args
                cimports.setdefault(module.value.to_python(), set()).update(v.to_python() for v in symbols.value)
            elif method == 'declare_attr':
                cls = registered_classes[name]
                cls.setdefault('attrs', odict())[args.value[0].value.to_python()] = args.value[1].value.to_python()
            elif method == 'register_property':
                cls = registered_classes[name]
                cls.setdefault('props', odict())[args.value[0].value.to_python()] = args.value[1].value.to_python()

            # Remove godot_cpp hints
            tree[i] = empty(tree[i])

        elif node.type == 'def' and node.decorators:
            for j, dec in enumerate(node.decorators):
                if str(dec.value.value[0]) not in directives or len(dec.value.value) != 3 or \
                    str(dec.value.value[2]) != 'register_method': continue
                name, _, method = dec.value.value
                cls = registered_classes[str(name)]
                return_type = returntype(node)
                args = {arg.target.value: argtype(arg) for arg in node.arguments}
                args['self'] = f'{name} *'
                cls.setdefault('methods', odict())[node.name] = odict(return_type=return_type, args=args)

                # Remove register_method decorator
                del tree[i].decorators[j]

    context = {
        'basename': basename,
        'cimports': cimports,
        'registered_classes': registered_classes
    }

    # from pprint import pprint
    # pprint(context)

    output = {
        f'{basename}__impl.py': tree.dumps(),
        f'{basename}__impl.pxd': Template(filename=os.path.join(templates_dir, 'impl.pxd.mako')).render(**context),
        f'{basename}__wrap.hpp': Template(filename=os.path.join(templates_dir, 'wrap.hpp.mako')).render(**context)
    }

    for fn, code in output.items():
        with open(os.path.join(output_dir, fn), 'w') as f:
            f.write(code)

    # Compile .py/.pxd, required for the next step
    cython_options = {}
    cython_options.update(default_options)
    cython_options['cplus'] = 1
    cython_options['output_file'] = os.path.join(output_dir, f'{basename}__impl.cpp')
    cython_options['language_level'] = 3
    cython_compile([os.path.join(output_dir, f'{basename}__impl.py')], CompilationOptions(cython_options))

    context['method_map'] = map_compiled_methods(registered_classes, output_dir, basename)
    wrapper_code = Template(filename=os.path.join(templates_dir, 'wrap.cpp.mako')).render(**context)
    with open(os.path.join(output_dir, f'{basename}__wrap.cpp'), 'w') as f:
            f.write(wrapper_code)


@click.command()
@click.argument('sourcefile', nargs=-1, type=click.File('r'))
@click.option('--output-dir', '-o')
@click.option('--version', '-V', is_flag=True)
def compilecpp(sourcefile, **opts):
    if (opts['version']):
        print(f'pygdn v{get_version()}')
    else:
        if not len(sourcefile):
            click.echo('No source files given')
            sys.exit(1)

        output_dir = opts.get('output_dir')

        for src in sourcefile:
            default_output_dir, filename = os.path.split(os.path.realpath(src.name))
            name, ext = os.path.splitext(filename)

            if not output_dir:
                output_dir = default_output_dir
            elif not os.path.isdir(output_dir):
                click.echo(f'Output directory "{output_dir}" does not exist')
                sys.exit(1)

            tree = RedBaron(src.read())
            compile(tree, output_dir, name)

pygodot.add_command(compilecpp)

def map_compiled_methods(classes, output_dir, basename):
    method_map = {}
    for cls in classes.values():
        for method in cls['methods']:
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
