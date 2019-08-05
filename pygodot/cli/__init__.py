import sys
import os
import click

from .binding_generator import generate, write_api_pxd

# root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-V', is_flag=True)
def pygodot(ctx, version):
    cmdname = os.path.basename(sys.argv[0])

    if version or not ctx.invoked_subcommand:
        from .. import get_version
        click.echo(f'{cmdname} v{get_version()}')

        if not version:
            click.echo(f'Usage: {cmdname} <command>')

        sys.exit(0)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-V', is_flag=True)
def binding_generator(ctx, version):
    cmdname = os.path.basename(sys.argv[0])

    if version or not ctx.invoked_subcommand:
        from .. import get_version
        click.echo(f'{cmdname} v{get_version()}')

        if not version:
            click.echo(f'Usage: {cmdname} <command>')

        sys.exit(0)


@binding_generator.add_command
@click.command()
def api():
    write_api_pxd(echo=click.echo)


@binding_generator.add_command
@click.command()
@click.option('--cpp/--no-cpp', default=True)
@click.option('--cython/--no-cython', default=True)
@click.option('--python/--no-python', default=True)
def classes(cpp, cython, python):
    generate(generate_cpp=cpp, generate_cython=cython, generate_python=python, echo=click.echo)


# Not used
# def install_dependencies(project_root, new_lib_root, symlink, force):
#     project_files = os.listdir(project_root)
#     if 'dist' not in project_files:
#         click.echo('Please run py2app/py2exe command to collect Python dependencies')
#         return False

#     # macOS-only section
#     plugin_dirs = glob.glob(os.path.join(project_root, 'dist', '*.plugin'))

#     if not plugin_dirs or not os.path.isdir(os.path.join(plugin_dirs[0], 'Contents', 'Resources')):
#         click.echo(f'Collected plugin was not found at "{project_root}"')
#         return False

#     resource_dir = os.path.join(plugin_dirs[0], 'Contents', 'Resources')
#     # end macOS

#     target_resource_dir = os.path.join(new_lib_root, '_pygodot.env')
#     print(resource_dir, target_resource_dir)

#     # Run only if there is no target_resource_dir or with a --force flag
#     if force and os.path.exists(target_resource_dir):
#         if not os.path.islink(target_resource_dir):
#             click.echo(f'Please remove "{target_resource_dir}" and try again')
#             return False
#         # Remove only symlinks, not full directory trees!
#         click.echo(f'Removing old destination resources "{target_resource_dir}"')
#         os.unlink(target_resource_dir)
#     if not os.path.exists(target_resource_dir):
#         if symlink:
#             os.symlink(resource_dir, target_resource_dir)
#         else:
#             shutil.copytree(resource_dir, target_resource_dir)
#     return True


# def ensure_godot_project_path(path):
#     godot_root = detect_godot_project(path)

#     if not godot_root:
#         click.echo('No Godot project detected')
#         sys.exit(1)

#     return os.path.realpath(godot_root)


# def detect_godot_project(dir, fn='project.godot'):
#     if not dir or not fn:
#         return

#     if os.path.isdir(dir) and 'project.godot' in os.listdir(dir):
#         return dir

#     return detect_godot_project(*os.path.split(dir))
