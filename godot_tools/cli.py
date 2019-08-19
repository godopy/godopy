import sys
import os
import click

from .binding_generator import generate, write_api_pxd

USE_INTERNAL_CYTHON = True
HERE = os.path.abspath(os.path.dirname(__file__))


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


# @click.command()
# @click.argument('source')
# @click.argument('outfile', type=click.File('w'), default=sys.stdout)
# def pygodot_cython(source, outfile):
#     pyinit_src_symbol = 'PyInit_' + os.path.basename(outfile.name[:-4])
#     pyinit_dst_symbol = 'PyInit_' + strip_internal_path(outfile.name).replace(os.sep, '__')[:-4]

#     tempfile = outfile.name.replace('.cpp', '.temp.cpp')
#     tempheaderfile = outfile.name.replace('.cpp', '.temp.h')

#     header_path = outfile.name.replace('.cpp', '.hpp')

#     if USE_INTERNAL_CYTHON:
#         internal_pyenv = os.path.realpath(os.path.join(HERE, '..', 'buildenv'))

#         if sys.platform == 'win32':
#             os.environ['PYTHONPATH'] = ';'.join([
#                 os.path.join(internal_pyenv, '..', 'deps', 'python', 'PCBuild', 'amd64'),
#                 os.path.join(internal_pyenv, 'Lib', 'site-packages')
#             ])
#             # python_exe = os.path.join(internal_pyenv, 'Scripts', 'python3')
#             cython_exe = os.path.join(internal_pyenv, 'Scripts', 'cython')
#         else:
#             os.environ['PYTHONPATH'] = ':'.join([
#                 os.path.join(internal_pyenv, 'lib', 'python3.8'),
#                 os.path.join(internal_pyenv, 'lib', 'python3.8', 'site-packages'),
#                 os.path.join(internal_pyenv, 'lib', 'python3.8', 'lib-dynload')
#             ])
#             # python_exe = os.path.join(internal_pyenv, 'bin', 'python3')
#             cython_exe = os.path.join(internal_pyenv, 'bin', 'cython')

#         try:
#             subprocess.run([cython_exe, '--fast-fail', '-3', '--cplus', '-o', tempfile, source], check=True)
#         except subprocess.CalledProcessError:
#             raise SystemExit('Cython compilation finished with errors')
#     else:
#         from Cython.Compiler import Options
#         from Cython.Compiler.Main import compile

#         directives = {
#             'c_string_encoding': 'utf-8'
#         }

#         options = Options.CompilationOptions(Options.default_options, compiler_directives=directives)

#         Options.fast_fail = True
#         options.output_file = tempfile
#         options.cplus = 1
#         options.language_level = 3

#         result = compile(source, options)
#         if result.num_errors > 0:
#             raise SystemExit('Cython compilation finished with errors')

#     def clean_line(line):
#         if pyinit_src_symbol in line:
#             line = line.replace(pyinit_src_symbol, pyinit_dst_symbol)

#         # Fix variable declarations with GDCALLINGCONV, GDCALLINGCONV is valid only for functions
#         if line.lstrip().startswith('GDCALLINGCONV_'):
#             line = re.sub(r'^(\s+)(GDCALLINGCONV_VOID_PTR)(\s\w+;)$', r'\1void *\3', line)
#             line = re.sub(r'^(\s+)(GDCALLINGCONV_VOID)(\s\w+;)$', r'\1void\3', line)
#             line = re.sub(r'^(\s+)(GDCALLINGCONV_BOOL)(\s\w+;)$', r'\1bool\3', line)
#         return line

#     with open(tempfile, 'r', encoding='utf-8') as fp:
#         for line in fp:
#             outfile.write(clean_line(line))

#     os.unlink(tempfile)

#     if os.path.exists(tempheaderfile):
#         with open(header_path, 'w', encoding='utf-8') as outheaderfile:
#             with open(tempheaderfile, 'r', encoding='utf-8') as fp:
#                 for line in fp:
#                     outheaderfile.write(clean_line(line))

#         os.unlink(tempheaderfile)


# def strip_internal_path(path):
#     if is_internal_path(path):
#         components = path.split(os.sep)
#         return os.sep.join(components[1:])
#     return path


def is_internal_path(path):
    return path.startswith('internal-packages') or path.startswith('src')


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
