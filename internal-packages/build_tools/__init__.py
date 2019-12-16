import sys
import os
import re


def pygodot_cython():
    source, outfile_path = sys.argv[1:]
    pyinit_src_symbol = 'PyInit_' + os.path.basename(outfile_path[:-4])
    pyinit_dst_symbol = 'PyInit_' + strip_internal_path(outfile_path).replace(os.sep, '__')[:-4]

    tempfile = outfile_path.replace('.cpp', '.temp.cpp')
    tempheaderfile = outfile_path.replace('.cpp', '.temp.h')

    header_path = outfile_path.replace('.cpp', '.hpp')

    from Cython.Compiler import Options
    from Cython.Compiler.Main import compile, default_options, CompilationOptions

    directives = {
        'c_string_encoding': 'utf-8'
    }

    options = CompilationOptions(default_options, compiler_directives=directives)

    Options.fast_fail = True
    options.output_file = tempfile
    options.cplus = 1
    options.language_level = 3

    result = compile(source, options)
    if result.num_errors > 0:
        raise SystemExit('Cython compilation finished with errors')

    def clean_line(line):
        if pyinit_src_symbol in line:
            line = line.replace(pyinit_src_symbol, pyinit_dst_symbol)

        # Undefined by #define NO_IMPORT_ARRAY
        if '_import_array()' in line:
            line = line.replace('_import_array()', '0')

        # Fix variable declarations with GDCALLINGCONV, GDCALLINGCONV is valid only for functions
        if line.lstrip().startswith('GDCALLINGCONV_'):
            line = re.sub(r'^(\s+)(GDCALLINGCONV_VOID_PTR)(\s\w+;)$', r'\1void *\3', line)
            line = re.sub(r'^(\s+)(GDCALLINGCONV_VOID)(\s\w+;)$', r'\1void\3', line)
            line = re.sub(r'^(\s+)(GDCALLINGCONV_BOOL)(\s\w+;)$', r'\1bool\3', line)
        return line

    with open(outfile_path, 'w', encoding='utf-8') as outfile:
        with open(tempfile, 'r', encoding='utf-8') as fp:
            for line in fp:
                outfile.write(clean_line(line))

    os.unlink(tempfile)

    if os.path.exists(tempheaderfile):
        with open(header_path, 'w', encoding='utf-8') as outheaderfile:
            with open(tempheaderfile, 'r', encoding='utf-8') as fp:
                for line in fp:
                    outheaderfile.write(clean_line(line))

        os.unlink(tempheaderfile)


def strip_internal_path(path):
    if is_internal_path(path):
        components = path.split(os.sep)
        return os.sep.join(components[1:])
    return path


def is_internal_path(path):
    return path.startswith('internal-packages') or path.startswith('src')
