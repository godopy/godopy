import os
import sys
import json
import click

from redbaron import RedBaron

from . import get_version
from Cython.Compiler.Main import CompilationOptions, default_options

CPPDEFS_MODULE = 'godot_cpp'

api_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'godot_headers', 'api.json'))
with open(api_path, 'r') as f:
    gdapi = {i['name']: i for i in json.load(f)}

@click.group()
def pygodot():
    pass

def compile(tree, output_dir, name, mode='py'):
    macro_prefix = ''
    directives = set()
    cimports = {}  # TODO: Use ordered dict
    registered_classes = {}
    basename = name

    for i, node in enumerate(tree):
        if node.type == 'import' and joinval(node.value[0]) == CPPDEFS_MODULE:
            macro_prefix = node.value[0].target or joinval(node.value[0])
            directives.add(macro_prefix)
            # tree[i] = empty(tree[i])
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

            # print(name, method, args)
            tree[i] = empty(tree[i])
            if str(method) == 'register_class':
                name = str(targets[0])
                directives.add(name)

                base = args[1].value.to_python()

                # methods = {m['name']: m for m in gdapi[base]['methods']}
                # print(gdapi[base]['methods'])
                registered_classes[name] = {
                    'name': args[0].value.to_python(),
                    'base': base
                }

                cimports.setdefault('godot_cpp.gen', set()).add(base)
            elif str(method) == 'cimport':
                module, symbols = args
                cimports.setdefault(module.value.to_python(), set()).update(v.to_python() for v in symbols.value)
            elif str(method) == 'declare_attr':
                cls = registered_classes[str(name)]
                cls.setdefault('attrs', {})[args.value[0]] = args.value[1]
            elif str(method) == 'register_property':
                cls = registered_classes[str(name)]
                cls.setdefault('props', {})[args.value[0]] = args.value[1]
        elif node.type == 'def' and node.decorators:
            for j, dec in enumerate(node.decorators):
                if str(dec.value.value[0]) not in directives or len(dec.value.value) != 3 or \
                    str(dec.value.value[2]) != 'register_method': continue
                name, _, method = dec.value.value
                cls = registered_classes[str(name)]
                return_type = returntype(node)
                args = {arg.target.value: argtype(arg) for arg in node.arguments}
                args['self'] = f'{name} *'
                cls.setdefault('methods', {})[node.name] = {
                    'return_type': return_type,
                    'args': args
                }

                del tree[i].decorators[j] # RedBaron('@cython.cfunc\ndef dummpy(): pass')[0].decorators[0]

    print(cimports)
    print(registered_classes)

    with open(os.path.join(output_dir, f'{name}__x.py'), 'w') as f:
        f.write(tree.dumps())

    print('\nNOT IMPLEMENTED:'),
    print(f'Automatic generation of "{name}__x.pxd", "{name}__w.hpp" and "{name}__w.cpp" files.')

@click.command()
@click.argument('sourcefile', nargs=-1, type=click.File('r'))
@click.option('--output-dir', '-o')
@click.option('--cpp', is_flag=True, default=True, help="Compile expanded source to C++ with Cython")
@click.option('--version', '-V', is_flag=True)
def compilecpp(sourcefile, **opts):
    if (opts['version']):
        print(f'pygdn v{get_version()}')
    else:
        if not len(sourcefile):
            click.echo('No source files given')
            sys.exit(1)

        sources = list(reversed(sourcefile))
        src = sources.pop()

        default_output_dir, filename = os.path.split(os.path.realpath(src.name))
        name, ext = os.path.splitext(filename)

        output_dir = opts.get('output_dir')

        if not output_dir:
            output_dir = default_output_dir
        elif not os.path.isdir(output_dir):
            click.echo(f'Output directory "{output_dir}" does not exist')
            sys.exit(1)

        mode = 'cpp' if opts['cpp'] else 'py'

        tree = RedBaron(src.read())
        compile(tree, output_dir, name, mode)

pygodot.add_command(compilecpp)


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
