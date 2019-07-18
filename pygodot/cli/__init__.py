import sys
import os
import re
import glob
import json
import shutil

import click
from mako.template import Template

from .pxd_writer import PxdWriter, parse as parse_c_header
from ..cpp_interop.compiler import compile
from ..binding_generator import generate

pygodot_lib_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
templates_dir = os.path.join(pygodot_lib_root, 'pygodot', 'cli', 'templates')

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

def install_dependencies(project_root, new_lib_root, symlink, force):
    project_files = os.listdir(project_root)
    if 'dist' not in project_files:
        click.echo('Please run py2app/py2exe command to collect Python dependencies')
        return False

    #### macOS-only section
    plugin_dirs = glob.glob(os.path.join(project_root, 'dist', '*.plugin'))

    if not plugin_dirs or not os.path.isdir(os.path.join(plugin_dirs[0], 'Contents', 'Resources')):
        click.echo(f'Collected plugin was not found at "{project_root}"')
        return False

    resource_dir = os.path.join(plugin_dirs[0], 'Contents', 'Resources')
    ### end macOS

    target_resource_dir = os.path.join(new_lib_root, '_pygodot.env')
    print(resource_dir, target_resource_dir)

    # Run only if there is no target_resource_dir or with a --force flag
    if force and os.path.exists(target_resource_dir):
        if not os.path.islink(target_resource_dir):
            click.echo(f'Please remove "{target_resource_dir}" and try again')
            return False
        # Remove only symlinks, not full directory trees!
        click.echo(f'Removing old destination resources "{target_resource_dir}"')
        os.unlink(target_resource_dir)
    if not os.path.exists(target_resource_dir):
        if symlink:
            os.symlink(resource_dir, target_resource_dir)
        else:
            shutil.copytree(resource_dir, target_resource_dir)
    return True

@pygodot.add_command
@click.command()
@click.argument('path', required=True, type=click.Path(exists=False))
@click.option('--export', is_flag=True)
@click.option('--force', is_flag=True)
def install(path, export, force):
    symlink = not export
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

    project_root = ensure_pygodot_project_path(godot_root)
    if export:
        install_dependencies(project_root, new_lib_root, symlink, force)
    else:
        dst_env_dir = os.path.join(new_lib_root, '_pygodot.env')
        dst_pylib_dir = os.path.join(dst_env_dir, 'lib', 'python3.7')
        if not os.path.exists(dst_pylib_dir):
            os.makedirs(dst_pylib_dir)
        dst_sitepackages_dir = os.path.join(dst_env_dir, 'lib', 'python3.7', 'site-packages')
        print(dst_pylib_dir, dst_sitepackages_dir)
        src_pylib_dir = None
        src_sitepackages_dir = None

        for path in reversed(sys.path):
            if path.endswith('python3.7'):
                src_pylib_dir = path
            elif path.endswith('site-packages'):
                src_sitepackages_dir = path
        print(src_pylib_dir, src_sitepackages_dir)

        for fn in os.listdir(src_pylib_dir):
            if fn == 'site-packages':
                continue
            src_path = os.path.join(src_pylib_dir, fn)
            dst_path = os.path.join(dst_pylib_dir, fn)
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)

        if not os.path.exists(dst_sitepackages_dir):
            os.symlink(src_sitepackages_dir, dst_sitepackages_dir)


    libname = None
    for fn in os.listdir(pygodot_lib_root):
        if fn.startswith('_pygodot.cpython-'):
            libname = fn
        if fn.startswith('_pygodot.cpython-') or fn.startswith('lib_pygodot.cpython-'):
            src_path = os.path.join(pygodot_lib_root, fn)
            dst_path = os.path.join(new_lib_root, fn)
            if os.path.exists(dst_path):
                continue

                # click.echo(f'Removing old destination library file "{dst_path}"')
                # os.unlink(dst_path)
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
    godot_root = detect_godot_project(path)

    if not godot_root:
        click.echo('No Godot project detected')
        sys.exit(1)

    return os.path.realpath(godot_root)

def detect_godot_project(dir, fn='project.godot'):
    if not dir or not fn:
        return

    if os.path.isdir(dir) and 'project.godot' in os.listdir(dir):
        return dir

    return detect_godot_project(*os.path.split(dir))

def ensure_pygodot_lib_path(path):
    libfiles = glob.glob(os.path.join(path, 'pygodot.gdnlib'))

    if not libfiles:
        click.echo('No PyGodot library project detected')
        sys.exit(1)

    return os.path.realpath(libfiles.pop())

def ensure_pygodot_project_path(path):
    project_root = detect_pygodot_project(path)

    if not project_root:
        click.echo('No PyGodot project detected')
        sys.exit(1)

    return os.path.realpath(project_root)

def detect_pygodot_project(dir, fn='gdlibrary.py'):
    if not dir or not fn:
        return

    if os.path.isdir(dir) and 'gdlibrary.py' in os.listdir(dir):
        return dir

    return detect_pygodot_project(*os.path.split(dir))

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
def genapi():
    output_dir = os.path.join(pygodot_lib_root, 'pygodot', 'headers')

    if not os.path.isdir(output_dir):
        click.echo(f'"{output_dir}" does not exist. Something went wrong…')
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

    with open('gdnative_api.pxd', 'w', encoding='utf-8') as f:
        f.write(pxd)
    with open('__init__.py', 'w', encoding='utf-8') as f:
        pass

    pythonize_gdnative_api(output_dir)


def pythonize_gdnative_api(output_dir):
    from pprint import PrettyPrinter
    from collections import OrderedDict

    click.echo('Converting\n\tapi.json -> api.py\n'
        f'inside "{output_dir}" directory\n')

    inpath = os.path.join(output_dir, 'api.json')
    if not os.path.exists(inpath):
        click.echo(f'Required "api.json" file doesn\'t exist in "{output_dir}"')
        sys.exit(1)

    with open(inpath, encoding='utf-8') as fp:
        api = json.load(fp)

    pythonized = {}

    for entry in api:
        name = entry.pop('name')
        assert name not in pythonized

        pythonized[name] = entry
        for collection in ('properties', 'signals', 'methods', 'enums'):
            pythonized[name][collection] = {prop.pop('name'): prop for prop in entry[collection]}

    pp = PrettyPrinter(indent=1, compact=True, width=120)
    with open('api.py', 'w', encoding='utf-8') as fp:
        fp.write('CLASSES = %s\n' % pp.pformat(pythonized))

@pygodot.add_command
@click.command()
def genbindings():
    generate(pygodot_lib_root, click.echo)
