import os
import sys
import shutil
import zipfile

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_scripts import install_scripts
from setuptools.command.develop import develop

from distutils import log

if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    raise SystemExit("PyGodot requires Python version 3.6 or above.")


if os.path.realpath(os.path.dirname(__file__)) != os.path.realpath(os.getcwd()):
    os.chdir(os.realpath(os.path.dirname(__file__)))


class GodoPyExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


get_godot_exe = __import__('godot_tools', {}, {}, ['utils']).utils.get_godot_executable


class InstallScriptsAndGodotExecutables(install_scripts):
    def run(self):
        super().run()

        godot_exe = get_godot_exe()
        godot_exe_gui = get_godot_exe(noserver=True)

        log.info('Installing Godot executable from %r' % godot_exe_gui)

        target = os.path.join(self.install_dir, 'godot')
        if not self.dry_run:
            shutil.copy2(godot_exe_gui, target)
        self.outfiles.append(target)

        if godot_exe != godot_exe_gui:
            log.info('Installing Godot server executable from %r' % godot_exe)
            target = os.path.join(self.install_dir, 'godot_server')
            if not self.dry_run:
                shutil.copy2(godot_exe_gui, target)
            self.outfiles.append(target)


class GodoPyDevelop(develop):
    def install_wrapper_scripts(self, dist):
        if dist.project_name == 'godopy':
            godot_exe = get_godot_exe()
            godot_exe_gui = get_godot_exe(noserver=True)

            log.info('Installing Godot executable from %r' % godot_exe_gui)

            # FIXME: Make symlinks?

            target = os.path.join(self.script_dir, 'godot')
            self.add_output(target)
            if not self.dry_run:
                shutil.copy2(godot_exe_gui, target)

            if godot_exe != godot_exe_gui:
                log.info('Installing Godot server executable from %r' % godot_exe)

                target = os.path.join(self.script_dir, 'godot_server')
                self.add_output(target)
                if not self.dry_run:
                    shutil.copy2(godot_exe_gui, target)

        super().install_wrapper_scripts(dist)


class BuildSconsAndPackInnerPython(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_and_zip(ext)

    def build_and_zip(self, ext):
        cwd = os.getcwd()
        extension_path = self.get_ext_fullpath(ext.name)
        if extension_path.startswith(cwd):
            extension_path = extension_path[len(cwd):].lstrip(os.sep)

        ext_basepath, ext_fn = os.path.split(extension_path)
        ext_fn_parts = ext_fn.split('.')

        lib_fn = '%s.%s' % (ext_fn_parts[0], ext_fn_parts[-1])
        lib_path = os.path.join(self.build_temp, lib_fn)

        if self.dry_run:
            return

        scons = os.path.join(sys.prefix, 'Scripts', 'scons') if sys.platform == 'win32' else 'scons'
        args = [scons, 'shared_target=%s' % lib_path]

        self.spawn(args)

        with zipfile.ZipFile(extension_path, 'w', zipfile.ZIP_DEFLATED) as _zip:
            assert os.path.exists(lib_path), lib_path

            print('WRITE', lib_fn)
            with _zip.open(lib_fn, 'w') as _lib_dst:
                with open(lib_path, 'rb') as _lib_src:
                    _lib_dst.write(_lib_src.read())


version = __import__('godot_tools').__version__

packages = ['godot_tools', 'godot_tools.binding_generator', 'godot_tools.setup']
package_data = {
    'godot_tools.setup': ['templates/*.mako'],
    'godot_tools.binding_generator': ['templates/*.mako']
}

entry_points = {'console_scripts': ['godopy=godot_tools.cli:godopy', 'bindgen=godot_tools.cli:bindgen']}

install_requires = [
    'Mako',
    'scons',
    'Click'
]

setup_requires = [
    'scons',
    'Mako',
    'pycparser',
    'autopxd2'
]

setup(
    name='godopy',
    version=version,
    python_requires='>=3.6',
    packages=packages,
    package_data=package_data,
    cmdclass={
        'build_ext': BuildSconsAndPackInnerPython,
        'install_scripts': InstallScriptsAndGodotExecutables,
        'develop': GodoPyDevelop
    },
    ext_modules=[GodoPyExtension('_godopy')],
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points=entry_points
)
