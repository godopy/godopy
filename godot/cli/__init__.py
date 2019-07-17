import sys
import os
import re
import glob
import shutil

import click
from mako.template import Template

from .pxd_writer import PxdWriter, parse as parse_c_header
from godot.cpp_interop.compiler import compile

pygodot_lib_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
templates_dir = os.path.join(pygodot_lib_root, 'godot', 'cli', 'templates')

@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-V', is_flag=True)
def pygodot(ctx, version):
    if version or not ctx.invoked_subcommand:
        from .. import get_version
        click.echo(f'pygodot v{get_version()}')

        if not version:
            click.echo('Usage: pygodot <command>')

        sys.exit(0)

@pygodot.add_command
@click.command()
@click.argument('path', required=True, type=click.Path(exists=False))
@click.option('--symlink', '-S', is_flag=True)
def install(path, symlink):
    godot_root = ensure_godot_project_path(path)
    new_lib_root = os.path.realpath(path)

    if not os.path.exists(new_lib_root):
        os.makedirs(new_lib_root)
    if not os.path.isdir(new_lib_root):
        click.echo(f'{path} is not a directory')
        sys.exit(1)

    platform = None
    if sys.platform == 'darwin':
        platform = 'OSX.64'
    else:
        raise NotImplementedError('Only macOS platform is supported at the moment!')

    libname = None
    for fn in os.listdir(pygodot_lib_root):
        if fn.startswith('pygodot.cpython-'):
            libname = fn
        if fn.startswith('pygodot.cpython-') or fn.startswith('libpygodot.cpython-'):
            src_path = os.path.join(pygodot_lib_root, fn)
            dst_path = os.path.join(new_lib_root, fn)
            if os.path.exists(dst_path):
                # TODO: If the files are the same, just skip
                click.echo(f'Removing old destination library file "{dst_path}"')
                os.unlink(dst_path)
            if symlink:
                os.symlink(src_path, dst_path)
            else:
                shutil.copyfile(src_path, dst_path)
                shutil.copymode(src_path, dst_path)

    if not libname:
        click.echo('No GDNative libraries detected. Looks like PyGodot wasn\'t installed correctly')
        syhs.exit(1)

    pygodot_gdnlib = os.path.join(new_lib_root, 'pygodot.gdnlib')
    gdnlib_template = Template(filename=os.path.join(templates_dir, 'gdnlib.mako'))

    lib_path = os.path.join(new_lib_root, libname)
    gdnlib_resource = lib_path.replace(godot_root, '').lstrip(os.sep).replace(os.sep, '/')

    with open(pygodot_gdnlib, 'w', encoding='utf-8') as f:
        f.write(gdnlib_template.render(
            singleton=False,
            load_once=True,
            symbol_prefix='godot_',
            reloadable=False,
            libraries={platform: gdnlib_resource},
            dependencies={platform: ''}
        ))

@pygodot.add_command
@click.command()
@click.argument('path', required=True, type=click.Path(exists=True))
@click.argument('classname', required=True)
def installscript(path, classname):
    godot_root = ensure_godot_project_path(path)
    lib_path = ensure_pygodot_lib_path(path)

    gdnlib_resource = lib_path.replace(godot_root, '').lstrip(os.sep).replace(os.sep, '/')

    script_gdns = os.path.join(os.path.dirname(lib_path), resname(classname) + '.gdns')
    gdns_template = Template(filename=os.path.join(templates_dir, 'gdns.mako'))

    with open(script_gdns, 'w', encoding='utf-8') as f:
        f.write(gdns_template.render(
            gdnlib_resource=gdnlib_resource,
            classname=classname
        ))


def ensure_godot_project_path(path):
    godot_root = os.path.realpath(detect_godot_project(path))

    if not godot_root:
        click.echo('No Godot project detected')
        sys.exit(1)

    return godot_root

def detect_godot_project(dir, fn='project.godot'):
    if not dir or not fn:
        return

    if os.path.isdir(dir) and 'project.godot' in os.listdir(dir):
        return dir

    return detect_godot_project(*os.path.split(dir))

def ensure_pygodot_lib_path(path):
    libfiles = glob.glob(os.path.join(path, 'pygodot*'))

    if not libfiles:
        click.echo('No PyGodot library project detected')
        sys.exit(1)

    return os.path.realpath(libfiles.pop())

def resname(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

@pygodot.add_command
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


@pygodot.add_command
@click.command()
@click.option('--output-dir', '-o', default='./godot/headers')
def genapi(output_dir):
    if not os.path.isdir(output_dir):
        click.echo(f'"{output_dir}" does not exist. Please provide existing directory with Godot header files')
        sys.exit(1)

    click.echo('Converting\n\tgdnative_api_struct.gen.h -> gdnative_api.pxd\n'
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

    with open('gdnative_api.pxd', 'w') as f:
        f.write(pxd)
    with open('__init__.py', 'w') as f:
        pass
