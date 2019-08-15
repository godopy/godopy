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
        from .version import get_version
        click.echo(f'{cmdname} v{get_version()}')

        if not version:
            click.echo(f'Usage: {cmdname} <command>')

        sys.exit(0)


@click.command()
@click.argument('source')
@click.argument('outfile', type=click.File('w'), default=sys.stdout)
def pygodot_cython(source, outfile):
    pyinit_src_symbol = 'PyInit_' + os.path.basename(outfile.name[:-4])
    pyinit_dst_symbol = 'PyInit_' + outfile.name.replace(os.sep, '__')[:-4]

    tempfile = outfile.name.replace('.cpp', '.temp.cpp')
    tempheaderfile = outfile.name.replace('.cpp', '.temp.h')

    header_path = outfile.name.replace('.cpp', '.hpp')

    from Cython.Compiler import Options
    from Cython.Compiler.Main import compile

    directives = {
        'c_string_encoding': 'utf-8'
    }

    options = Options.CompilationOptions(Options.default_options, compiler_directives=directives)

    Options.fast_fail = True
    options.output_file = tempfile
    options.cplus = 1
    options.language_level = 3

    result = compile(source, options)
    if result.num_errors > 0:
        raise SystemExit('Cython compilation finished with errors')

    with open(tempfile, 'r', encoding='utf-8') as fp:
        for line in fp:
            if pyinit_src_symbol in line:
                line = line.replace(pyinit_src_symbol, pyinit_dst_symbol)
            outfile.write(line)

    os.unlink(tempfile)

    if os.path.exists(tempheaderfile):
        with open(header_path, 'w', encoding='utf-8') as outheaderfile:
            with open(tempheaderfile, 'r', encoding='utf-8') as fp:
                for line in fp:
                    if pyinit_src_symbol in line:
                        line = line.replace(pyinit_src_symbol, pyinit_dst_symbol)
                    outheaderfile.write(line)

        os.unlink(tempheaderfile)


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
