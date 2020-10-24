import os
import sys
import glob
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

from distutils import log

if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    raise SystemExit("GodoPy requires Python version 3.6 or above.")


if os.path.realpath(os.path.dirname(__file__)) != os.path.realpath(os.getcwd()):
    os.chdir(os.realpath(os.path.dirname(__file__)))


def get_godot_exe():
    godot_build_dir = os.environ.get('GODOT_BUILD')
    if not godot_build_dir:
        raise SystemExit("'GODOT_BUILD' environment variable is required.")

    godot_exe_list = crossplat_exe_glob(godot_build_dir, 'godot.*.64')

    if not godot_exe_list:
        raise SystemExit("Can't find Godot executable.")

    return godot_exe_list.pop()


def crossplat_exe_glob(godot_build_dir, pattern):
    if sys.platform == 'win32':
        pattern += '.exe'

    return glob.glob(os.path.join(godot_build_dir, 'bin', pattern))


class GodoPyExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


PYTHON_IGNORE = ('lib2to3', 'tkinter', 'ensurepip', 'parser', 'test', 'pip')
PYTHONLIB_FORMAT = 'xztar'
DEFAULT_VENV = 'env'


argv = []

venv = '.env'

for arg in sys.argv:
    if arg.startswith('venv'):
        venv = arg[5:]
    else:
        argv.append(arg)

sys.argv = argv


class BuildExtCommand(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_and_compress(ext)

    def build_and_compress(self, ext):
        cwd = os.getcwd()
        extension_path = self.get_ext_fullpath(ext.name)
        if extension_path.startswith(cwd):
            extension_path = extension_path[len(cwd):].lstrip(os.sep)

        ext_basepath, ext_fn = os.path.split(extension_path)
        python_basedir = os.path.join('3rdparty', 'python')

        if sys.platform == 'win32':
            python_exe = os.path.join(python_basedir, 'PCBuild', 'amd64', 'python.exe')
            python_lib = os.path.join(python_basedir, 'Lib')
            python_dynload = os.path.join(python_basedir, 'PCBuild', 'amd64')
            packages_dir = os.path.join(venv, 'Lib', 'site-packages')
        else:
            python_exe = os.path.join(python_basedir, 'build', 'bin', 'python3.9')
            python_lib = os.path.join(python_basedir, 'build', 'lib', 'python3.9')
            python_dynload = os.path.join(python_basedir, 'build', 'lib', 'python3.9', 'lib-dynload')
            packages_dir = os.path.join(venv, 'lib', 'python3.9', 'site-packages')

        cmd = [python_exe, '-c', "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, universal_newlines=True)
        ext_suffix = result.stdout.strip()

        lib_fn = '_godopy' + ext_suffix
        lib_path = os.path.join(self.build_temp, 'lib', lib_fn)

        if self.dry_run:
            return

        scons = os.path.join(sys.prefix, 'Scripts', 'scons') if sys.platform == 'win32' else 'scons'
        args = [scons, 'shared_target=%s' % lib_path, 'venv=%s' % venv]

        # Builds <temp>/_godopy.cpython-<version>-<platform>.<so|pyd> using Scons
        self.spawn(args)

        print_status = log._global_log.threshold > log.DEBUG

        def copy_python_lib_file(src, dst, fn, ratio, force=False):
            check_existance = True

            if dst.endswith('.py'):
                dst += 'c'
                if print_status:
                    sys.stdout.write('\r%d%%' % (ratio*100))
                    sys.stdout.flush()
                else:
                    log.debug('COMPILE %s -> %s' % (src, dst))
                changed = force or not os.path.exists(dst) or os.stat(src).st_mtime != os.stat(dst).st_mtime
                if not changed:
                    return

                cmd = [
                    python_exe,
                    '-c',
                    "from py_compile import compile; compile(%r, %r, dfile=%r, doraise=True)" % (src, dst, fn)
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    shutil.copystat(src, dst)
                except Exception as exc:
                    check_existance = False
                    log.error('Could not compile %r, error was: %r' % (dst, exc))
            else:
                if print_status:
                    sys.stdout.write('\r%d%%' % (ratio*100))
                    sys.stdout.flush()
                else:
                    log.debug('COPY %s -> %s' % (src, dst))
                changed = force or not os.path.exists(dst) or os.stat(src).st_mtime != os.stat(dst).st_mtime
                if not changed:
                    return

                target_dir = os.path.dirname(dst)
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)

                shutil.copy2(src, dst)

            if check_existance:
                assert os.path.exists(dst), dst

        log.info('preparing Python libraries')
        to_copy = []
        for dirname, subdirs, files in os.walk(python_lib):
            if not files or '__pycache__' in dirname:
                continue

            skip = False
            short_dirname = dirname.replace(python_lib, '').lstrip(os.sep)
            for skipdir in ('site-packages', 'lib-dynload', *PYTHON_IGNORE):
                if short_dirname.startswith(skipdir):
                    skip = True

            if skip:
                continue

            target_dirname = os.path.join(self.build_temp, 'lib', short_dirname)

            for fn in files:
                src = os.path.join(dirname, fn)
                dst = os.path.join(target_dirname, fn)
                to_copy.append((src, dst, fn))

        for fn in os.listdir(python_dynload):
            src = os.path.join(python_dynload, fn)
            dst = os.path.join(self.build_temp, 'lib', fn)
            to_copy.append((src, dst, fn))

        for dirname, subdirs, files in os.walk(packages_dir):
            if not files or '__pycache__' in dirname:
                continue

            target_dirname = os.path.join(self.build_temp, 'lib', dirname.replace(packages_dir, '').lstrip(os.sep))

            for fn in files:
                src = os.path.join(dirname, fn)
                dst = os.path.join(target_dirname, fn)
                to_copy.append((src, dst, fn))

        for i, (src, dst, fn) in enumerate(to_copy):
            copy_python_lib_file(src, dst, fn, (i+1)/len(to_copy))
        if print_status:
            print()

        log.info('preparing auxillary Python executable')
        bin_dir = os.path.join(self.build_temp, 'bin')
        if not os.path.isdir(bin_dir):
            os.makedirs(bin_dir)
        shutil.copy(python_exe, os.path.join(bin_dir, 'python.exe'))

        archive_base_path, ext = os.path.splitext(extension_path)
        log.info('writing collected Python environment to %r' % extension_path)
        # Compress files from self.build_temp into a PYTHONLIB_FORMAT archive
        result = shutil.make_archive(archive_base_path, PYTHONLIB_FORMAT, self.build_temp)
        # Rename the archive to self.get_ext_fullpath(ext.name)
        shutil.move(result, extension_path)


# The name `godot_tools` was chosen to avoid any confusion with `godopy` and `godot` under `_lib`
# `godopy` and `godot` packages are designed to be accessed from GDNative modules.
# `godot_tools` is a meta package designed to build and/or package a customized Python interpreter
# inside the user's Godot project (and/or a generic Godot project in `script_runner/project`)
packages = [
    'godot_tools',
    'godot_tools.setup',
    'godot_tools.tests',
    'godot_tools.script_runner',
    'godot_tools.binding_generator'
]

package_data = {
    'godot_tools.setup': ['templates/*.mako'],
    'godot_tools.binding_generator': ['templates/*.mako']
}

setup(
    packages=packages,
    package_data=package_data,
    cmdclass={
        'build_ext': BuildExtCommand
    },
    ext_modules=[GodoPyExtension('_godopy')],
)
