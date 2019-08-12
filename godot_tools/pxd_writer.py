import sys
import subprocess

from pycparser import c_parser

from autopxd.nodes import Array
from autopxd.writer import AutoPxd
from autopxd.declarations import BUILTIN_HEADERS_DIR, IGNORE_DECLARATIONS


# Copied from autopxd2, modified preprocessor for macOS
def preprocess(code, extra_cpp_args=[]):
    preprocessor = ['gcc', '-E', '-std=c99']
    if sys.platform == 'darwin':
        preprocessor = ['clang', '-E', '-std=c99']
    elif sys.platform == 'win32':
        preprocessor = ['cpp.exe', '-std=c99']

    args = ['-nostdinc', '-D__attribute__(x)=', '-I', BUILTIN_HEADERS_DIR]
    proc = subprocess.Popen(preprocessor + args + extra_cpp_args + ['-'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = [proc.communicate(input=code.encode('utf-8'))[0]]
    while proc.poll() is None:
        result.append(proc.communicate()[0])
    if proc.returncode:
        raise Exception('Invoking C preprocessor failed')

    return b''.join(result).decode('ascii', 'ignore')


def parse(code, extra_cpp_args=[]):
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
