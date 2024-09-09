import os
import sys
import subprocess
from pathlib import Path

from pycparser import c_parser

from autopxd import Array, AutoPxd, BUILTIN_HEADERS_DIR, IGNORE_DECLARATIONS


BASE_DIR = Path(__file__).resolve().parent


# Copied from autopxd2, modified preprocessor for macOS
def preprocess(code, extra_cpp_args=[]):
    preprocessor = ['gcc', '-E', '-std=c99']
    if sys.platform == 'darwin':
        preprocessor = ['clang', '-E', '-std=c99']
    elif sys.platform == 'win32':
        preprocessor = ['gcc', '-std=c99', '-E']

    args = ['-nostdinc', '-D__attribute__(x)=', '-I', BUILTIN_HEADERS_DIR]
    proc = subprocess.Popen(preprocessor + args + extra_cpp_args + ['-'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = [proc.communicate(input=code.encode('utf-8'))[0]]
    while proc.poll() is None:
        result.append(proc.communicate()[0])
    if proc.returncode:
        raise Exception('Invoking C preprocessor failed')

    return b''.join(result).decode('ascii', 'ignore')


def parse_c_header(code, extra_cpp_args=[]):
    preprocessed = preprocess(code, extra_cpp_args=extra_cpp_args)
    stripped = []
    for line in preprocessed.splitlines():
        if not line.startswith('#'):
            stripped.append(line)
    stripped = ''.join(stripped)
    parser = c_parser.CParser()
    ast = parser.parse(stripped)
    decls = []
    for decl in ast.ext:
        if hasattr(decl, 'name') and decl.name not in IGNORE_DECLARATIONS:
            decls.append(decl)
    ast.ext = decls
    return ast


class PxdWriter(AutoPxd):
    def visit_ArrayDecl(self, node):
        dim = ''
        if hasattr(node, 'dim'):
            if hasattr(node.dim, 'value'):
                dim = node.dim.value
            elif hasattr(node.dim, 'name') and node.dim.name in self.constants:
                dim = str(self.constants[node.dim.name])
        self.dimension_stack.append(dim)
        decls = self.collect(node)
        # FIXME: Some '_dont_touch_that' arrays failed the assertion "assert len(decls) == 1"
        # in the original autopxd2 code
        # Invalid PXD code will be removed automatically
        self.append(Array(decls[0], self.dimension_stack))
        self.dimension_stack = []


stdint_declarations = [
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'
]


def write_api_pxd(echo=print):
    output_dir = BASE_DIR / 'src' / 'pyxlib'
    input_dir = BASE_DIR / 'godot-cpp' / 'gdextension'

    print(input_dir, output_dir)
    if not os.path.isdir(output_dir):
        echo(f'"{output_dir}" does not exist. Something went wrong…')
        sys.exit(1)

    echo("Converting 'gdextension_interface.h' -> 'gdextension_interface.pxd'")

    inpath =  input_dir / 'gdextension_interface.h'
    outpath = output_dir / 'gdextension_interface.pxd'
    if not os.path.exists(inpath):
        echo(f'Required "gdextension_interface.h" file doesn\'t exist in "{input_dir}"')
        sys.exit(1)

    # initial_dir = os.getcwd()

    # os.chdir(output_dir)

    with open(inpath, 'r') as infile:
        code = infile.read()

    extra_cpp_args = ['-I', '.']
    if sys.platform == 'darwin':
        # TODO: Use 'xcode-select -p' output
        extra_cpp_args += ['-I', "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include"]

    p = PxdWriter(inpath)
    p.visit(parse_c_header(code, extra_cpp_args=extra_cpp_args))

    pxd = 'from libc.stdint cimport {:s}\n'.format(', '.join(stdint_declarations))
    pxd += 'from libc.stddef cimport wchar_t\nfrom libcpp cimport bool\n'
    # pxd += 'ctypedef uint32_t char32_t\n'
    # pxd += 'ctypedef uint16_t char16_t\n'
    pxd += '\n'
    pxd += str(p)
    pxd = pxd.replace('uint8_t _dont_touch_that[]', 'pass')

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(pxd)

    # with open('__init__.py', 'w', encoding='utf-8') as f:
    #    pass

    # os.chdir(initial_dir)


if __name__ == '__main__':
    write_api_pxd()
