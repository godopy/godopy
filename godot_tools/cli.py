import sys
import os
import click

from .binding_generator import generate, write_api_pxd

USE_INTERNAL_CYTHON = True
HERE = os.path.abspath(os.path.dirname(__file__))


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', '-V', is_flag=True)
def godopy(ctx, version):
    cmdname = os.path.basename(sys.argv[0])

    if version or not ctx.invoked_subcommand:
        from .version import get_version
        click.echo(f'{cmdname} v{get_version()}')

        if not version:
            click.echo(f'Usage: {cmdname} <command>')

        sys.exit(0)


@godopy.add_command
@click.command()
@click.argument('path')
def new_project(path):
    if os.path.exists(path):
        raise SystemExit('%r already exists' % path)

    os.makedirs(path)

    # Create an empty file
    with open(os.path.join(path, 'project.godot'), 'w', encoding='utf-8'):
        pass


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
