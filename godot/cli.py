import sys
import os

import click
from .pxd_writer import PxdWriter, parse as parse_c_header
from godot_cpp.compiler import compile

@click.group(invoke_without_command=True)
@click.option('--version', '-V', is_flag=True)
def pygodot(version):
    if version:
        from . import get_version
        click.echo(f'pygodot v{get_version()}')
        sys.exit(0)


@click.command()
@click.argument('sourcefile', nargs=-1, type=click.File('r'))
@click.option('--output-dir', '-o')
def compilecpp(sourcefile, **opts):
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

        compile(src.read(), output_dir, name)

pygodot.add_command(compilecpp)


@click.command()
@click.option('--output-dir', '-o', default='./godot_headers')
def genapi(output_dir):
    if not os.path.isdir(output_dir):
        click.echo(f'"{output_dir}" does not exist. Please provide existing directory with Godot header files')
        sys.exit(1)

    click.echo('Converting\n\tgdnative_api_struct.gen.h -> gdnative_api_struct__gen.pxd\n'
        f'inside "{output_dir}" directory\n')

    inpath = os.path.join(output_dir, 'gdnative_api_struct.gen.h')
    if not os.path.exists(inpath):
        click.echo(f'Required "gdnative_api_struct.gen.h" file doesn\'t exist in "{output_dir}"')
        sys.exit(1)

    os.chdir(output_dir)
    fname = 'gdnative_api_struct.gen.h'

    with open(fname, 'r') as infile:
        code = infile.read()

    extra_cpp_args = ['-I', '.']
    if sys.platform == 'darwin':
        extra_cpp_args += ['-I', "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"]

    p = PxdWriter(fname)
    p.visit(parse_c_header(code, extra_cpp_args=extra_cpp_args))

    pxd = 'from libc.stdint cimport {:s}\n'.format(', '.join(p.stdint_declarations))
    pxd += 'from libc.stddef cimport wchar_t\nfrom libcpp cimport bool\n\n'
    pxd += str(p)
    pxd = pxd.replace('uint8_t _dont_touch_that[]', 'pass')
    pxd = pxd.replace('extern from "gdnative_api_struct.gen.h":', 'extern from "gdnative_api_struct.gen.h" nogil:')

    with open('gdnative_api_struct__gen.pxd', 'w') as f:
        f.write(pxd)
    with open('__init__.py', 'w') as f:
        pass

pygodot.add_command(genapi)
