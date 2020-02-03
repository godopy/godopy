import os
import sys
import click
import subprocess

from pathlib import Path
from importlib import import_module

from .binding_generator import generate, write_api_pxd

HERE = Path(__file__).resolve(strict=True).parents[0]

ENVIRONMENT_VARIABLE = 'GODOPY_PROJECT_MODULE'

GODOT_EXE = 'godot.exe' if sys.platform == 'win32' else 'godot'


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-V', is_flag=True)
def godopy(ctx, version):
    cmdname = os.path.basename(sys.argv[0])

    if version or not ctx.invoked_subcommand:
        from .version import get_version
        click.echo(f'GodoPy v{get_version()}')

        if not version:
            click.echo(f'Usage: {cmdname} <command>')

        sys.exit(0)


@godopy.add_command
@click.command()
@click.argument('path')
def newproject(path):
    if os.path.exists(path):
        raise SystemExit('%r already exists' % path)

    os.makedirs(path)

    # Create an empty file
    with open(os.path.join(path, 'project.godot'), 'w', encoding='utf-8'):
        pass


@godopy.add_command
@click.command()
@click.argument('script')
def runpy(script):
    project_module_pythonpath = os.environ.get(ENVIRONMENT_VARIABLE, 'godot_tools.script_runner.project')
    project_module = import_module(project_module_pythonpath)

    project_path = project_module.GODOT_PROJECT
    main_script = getattr(project_module, 'RUNPY_MAIN', 'Main')
    if not main_script.endswith('.gdns'):
        main_script += '.gdns'

    if not os.path.isfile(os.path.join(project_path, 'project.godot')):
        raise SystemExit('Please run "godopy enable-runpy" to enable "runpy" command.')

    if script.endswith('.py'):
        script = script[:-3]
    path = os.path.realpath(script.replace('.', os.sep))
    dirname, name = os.path.split(path)

    os.environ['SCRIPT_PATH'] = dirname
    os.environ['GODOPY_MAIN_MODULE'] = name

    cmd = [GODOT_EXE, '--path', project_path, '-s', main_script]
    subprocess.run(cmd, check=True)


@godopy.add_command
@click.command()
def test():
    project_module = import_module('godot_tools.script_runner.project')

    project_path = project_module.GODOT_PROJECT

    if not os.path.isfile(os.path.join(project_path, 'project.godot')):
        raise SystemExit('Please run "godopy enable-runpy" to enable "test" command.')

    os.environ['SCRIPT_PATH'] = str(project_module.BASE_DIR / 'tests')
    os.environ['GODOPY_MAIN_MODULE'] = 'runtests'

    cmd = [GODOT_EXE, '--path', project_path, '-s', 'Main.gdns']
    subprocess.run(cmd, check=True)


@godopy.add_command
@click.command()
def run():
    project_module_pythonpath = os.environ.get(ENVIRONMENT_VARIABLE)
    if not project_module_pythonpath:
        _raise_unconfigured_project(project_module_pythonpath)

    project_module = import_module(project_module_pythonpath)
    project_path = project_module.GODOT_PROJECT

    cmd = [GODOT_EXE, '--path', project_path]
    subprocess.run(cmd, check=True)


@godopy.add_command
@click.command()
def runeditor():
    project_module_pythonpath = os.environ.get(ENVIRONMENT_VARIABLE)
    if not project_module_pythonpath:
        _raise_unconfigured_project(project_module_pythonpath)

    project_module = import_module(project_module_pythonpath)
    project_path = project_module.GODOT_PROJECT

    cmd = [GODOT_EXE, '--path', project_path, '-e']
    subprocess.run(cmd, check=True)


@godopy.add_command
@click.command()
@click.argument('script')
def runscript(script):
    project_module_pythonpath = os.environ.get(ENVIRONMENT_VARIABLE)
    if not project_module_pythonpath:
        _raise_unconfigured_project(project_module_pythonpath)

    project_module = import_module(project_module_pythonpath)
    project_path = project_module.GODOT_PROJECT

    cmd = [GODOT_EXE, '--path', project_path, '-s', script]
    subprocess.run(cmd, check=True)


@godopy.add_command
@click.command()
@click.option('--force', is_flag=True)
def installscripts(force):
    project_module_pythonpath = os.environ.get(ENVIRONMENT_VARIABLE)
    if not project_module_pythonpath:
        _raise_unconfigured_project(project_module_pythonpath)

    mod = import_module(project_module_pythonpath)

    from godot_tools.setup import godot_setup, libraries, extensions

    sources = getattr(mod, 'NATIVESCRIPT_SOURCES', {})
    development_path = getattr(mod, 'DEVELOPMENT_PATH', None)

    libclass = sources and libraries.GDNativeLibrary or libraries.GenericGDNativeLibrary

    def make_ext(ext_path):
        return extensions.NativeScript(
            ext_path + '.gdns',
            class_name=os.path.basename(ext_path),
            sources=sources.get(ext_path, [])
        )

    save_argv = sys.argv[:]
    sys.argv = [sys.argv[0], 'install']
    if force:
        sys.argv.append('--force')

    kwargs = dict(
        godot_project=mod.GODOT_PROJECT,
        python_package=mod.PYTHON_PACKAGE,
        project_module=mod,
        library=libclass(mod.GDNATIVE_LIBRARY),
        extensions=[make_ext(ext) for ext in mod.NATIVESCRIPT_EXTENSIONS]
    )

    if development_path:
        kwargs['development_path'] = development_path

    if not os.path.exists(mod.GODOT_PROJECT / 'project.godot'):
        # Create an empty file
        with open(mod.GODOT_PROJECT / 'project.godot', 'w', encoding='utf-8'):
            pass

    godot_setup(**kwargs)

    sys.argv = save_argv


def _raise_unconfigured_project(name=None):
    desc = ('project %s' % name) if name else 'project'

    raise SystemExit("Requested %s, but project is not configured. "
                     "You must define the environment variable %s "
                     "before accessing project." % (desc, ENVIRONMENT_VARIABLE))


@godopy.add_command
@click.command()
def enable_runpy():
    import godot_tools

    from godot_tools.setup import godot_setup
    from godot_tools.setup.libraries import GenericGDNativeLibrary
    from godot_tools.setup.extensions import NativeScript

    os.chdir(godot_tools.__path__[0])
    project_dir = os.path.join('script_runner', 'project')

    if not os.path.isdir(project_dir):
        os.makedirs(project_dir)

    with open(os.path.join(project_dir, 'project.godot'), 'w', encoding='utf-8'):
        pass

    save_argv = sys.argv[:]
    sys.argv = [sys.argv[0], 'install', '--force']
    godot_setup(
        godot_project='script_runner/project',
        python_package='script_runner',
        development_path=os.getcwd(),
        library=GenericGDNativeLibrary('script-runner.gdnlib'),
        extensions=[
            NativeScript('Main.gdns', class_name='Main')
        ]
    )
    sys.argv = save_argv


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-V', is_flag=True)
def bindgen(ctx, version):
    cmdname = os.path.basename(sys.argv[0])

    if version or not ctx.invoked_subcommand:
        from .version import get_version
        click.echo(f'{cmdname} v{get_version()}')

        if not version:
            click.echo(f'Usage: {cmdname} <command>')

        sys.exit(0)


@bindgen.add_command
@click.command()
def api():
    write_api_pxd(echo=click.echo)


@bindgen.add_command
@click.command()
@click.option('--cpp/--no-cpp', default=True)
@click.option('--cython/--no-cython', default=True)
@click.option('--python/--no-python', default=True)
def classes(cpp, cython, python):
    generate(generate_cpp=cpp, generate_cython=cython, generate_python=python, echo=click.echo)
