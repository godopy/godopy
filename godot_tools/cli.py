import sys
import os
import click

from .binding_generator import generate, write_api_pxd


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
def bindgen(ctx, version):
    cmdname = os.path.basename(sys.argv[0])

    if version or not ctx.invoked_subcommand:
        from .. import get_version
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
