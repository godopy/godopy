import os
import sys
import math
import glob
import shutil
import hashlib
import zipfile
import subprocess

from distutils import log
from setuptools import Extension
from setuptools.command.build_ext import build_ext

from mako.template import Template

from ..version import get_version
from ..utils import is_internal_path

from .enums import ExtType

root_dir = os.getcwd()  # XXX
tools_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
templates_dir = os.path.join(tools_root, 'setup', 'templates')


class NativeScript(Extension):
    def __init__(self, name, *, addon_prefix=None, sources=None, python_sources=None, class_name=None):
        self._gdnative_type = ExtType.NATIVESCRIPT
        self._nativescript_classname = class_name
        self._dynamic_sources = python_sources or []

        self._addon_prefix = addon_prefix

        super().__init__(name, sources=(sources or []))


class Addon(Extension):
    def __init__(self, name, *, data_files=None, editor_only=True, **kwargs):
        self.python_package = name
        self._gdnative_type = ExtType.ADDON

        self._data_file_patterns = data_files or []
        self._editor_only = editor_only
        self._addon_metadata = kwargs

        super().__init__(name, sources=[])


# TODO: Make target configurable
build_target = 'release'

TOOLS_PACKAGES = ('Cython', 'IPython', 'ipython_genutils', 'jedi', 'parso', 'pexpect', 'traitlets', 'ptyprocess')


# TODO:
# * Allow users to exclude Python dependencies with glob patterns
# * Encapsulate Python packaging into a separate class and let users create custom classes
# * Optionally allow to compress all binary Python extensions to .pak files (uncompress to user:// at runtime)
class GDNativeBuildExt(build_ext):
    godot_project = None
    addons = {}
    gdnative_library_path = None
    generic_setup = False

    build_context = {
        '__version__': get_version(),
        'godot_headers_path': os.path.normpath(os.path.join(tools_root, '..', '_lib', 'godot_headers')),
        'godopy_bindings_path': os.path.dirname(tools_root),
        'singleton': False,
        'gdnative_options': False,
        'pyx_sources': [],
        'cpp_sources': []
    }

    godot_resources = {}
    godot_resource_files = {}
    python_dependencies = {}

    def run(self):
        dependencies_collected = False

        for ext in self.extensions:
            if self.godot_project:
                if not dependencies_collected:
                    log.info('collecting Python dependencies')
                    self.collect_dependencies()
                    dependencies_collected = True

                log.info('setting up GDNative {0}: {1}'.format(ext._gdnative_type.name.lower(), ext.name))

            getattr(self, 'collect_godot_{0}_data'.format(ext._gdnative_type.name.lower()))(ext)

        if self.generic_setup:
            self.run_copylib()
        else:
            # TODO: Check that required development files exist
            self.run_build()

        for res_path, content in self.godot_resources.items():
            self.write_target_file(os.path.join(self.godot_project.path_prefix, res_path), content,
                                   pretty_path='res://%s' % res_path, is_editable_resource=True)

        for res_path, source in self.godot_resource_files.items():
            target = os.path.join(self.godot_project.path_prefix, res_path)
            if os.path.exists(target) and os.stat(source).st_mtime == os.stat(target).st_mtime and not self.force:
                log.info('skip copying "%s"' % make_resource_path(self.godot_root, target))
                continue

            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir) and not self.dry_run:
                os.makedirs(target_dir)

            log.info('copying "%s"' % make_resource_path(self.godot_root, target))
            if not self.dry_run:
                shutil.copy2(source, target)

        self.package_dependencies()

        setup_script = self.gdnative_library_path.replace('.gdnlib', '__setup.gd')
        log.info('updating Godot project settings, running "res://%s"' % setup_script)
        if not self.dry_run:
            subprocess.run([
                'godot',
                '--path', self.godot_project.path_prefix,
                '-s', 'res://%s' % setup_script
            ], check=True)

        log.info('removing "res://%s"' % setup_script)
        if not self.dry_run:
            os.unlink(os.path.join(self.godot_project.path_prefix, setup_script))

    def run_copylib(self):
        # source_zip = os.path.join(self.build_context['godopy_bindings_path'], self.build_context['godopy_zip_name'])
        source = os.path.join('build', 'lib', self.build_context['godopy_library_name'])
        target = self.build_context['target']

        assert os.path.exists(source), source

        if os.path.exists(target) and os.stat(source).st_mtime == os.stat(target).st_mtime and not self.force:
            log.info('skip copying "res://%s"' % make_resource_path(self.godot_root, target))
            return

        target_dir = os.path.dirname(target)
        if not os.path.exists(target_dir) and not self.dry_run:
            os.makedirs(target_dir)

        log.info('copying "res://%s"' % make_resource_path(self.godot_root, target))
        if self.dry_run:
            return

        shutil.copy2(source, target)

    def run_build(self):
        cpp_lib_template = Template(filename=os.path.join(templates_dir, 'gdlibrary.cpp.mako'))
        cpp_lib_path = self.build_context['cpp_library_path']

        self.write_target_file(cpp_lib_path, cpp_lib_template.render(**self.build_context))

        self.build_context['cpp_sources'].append(make_relative_path(self.build_context['cpp_library_path']))

        scons_template = Template(filename=os.path.join(templates_dir, 'SConstruct.mako'))
        self.write_target_file('SConstruct', scons_template.render(**self.build_context))

        scons = os.path.join(sys.prefix, 'Scripts', 'scons') if sys.platform == 'win32' else 'scons'

        if not self.dry_run:
            self.spawn([scons, 'target=%s' % build_target])

    def write_target_file(self, path, content, pretty_path=None, is_editable_resource=False):
        if pretty_path is None:
            pretty_path = path

        if os.path.exists(path) and not self.force:
            # Check if there are any changes
            new_hash = hashlib.sha1(content.encode('utf-8'))
            with open(path, encoding='utf-8') as fp:
                old_hash = hashlib.sha1(fp.read().encode('utf-8'))
            if old_hash.digest() == new_hash.digest():
                log.info('skip writing "%s"' % pretty_path)
                return
            elif is_editable_resource and not self.force:
                # Do not overwrite user resources without --force flag
                log.warn('WARNING! modified resource already exists: "%s"' % pretty_path)
                return

        log.info('writing "%s"' % pretty_path)
        if not self.dry_run:
            dirname, basename = os.path.split(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(content)

    def package_dependencies(self):
        lib_root = os.path.join(self.godot_project.path_prefix, self.godot_project.binary_path, 'lib')
        libtools_root = os.path.join(self.godot_project.path_prefix, self.godot_project.binary_path, 'libtools')

        if self.dry_run:
            return

        if not os.path.isdir(lib_root):
            os.makedirs(lib_root)

        if not os.path.isdir(libtools_root):
            os.makedirs(libtools_root)

        for dirname, subdirs, files in os.walk(self.python_dependencies['lib_dir']):
            base_dirname = dirname.replace(self.python_dependencies['lib_dir'], '').lstrip(os.sep)

            root = lib_root

            for tools_pkg in TOOLS_PACKAGES:
                if base_dirname.startswith(tools_pkg):
                    root = libtools_root
                    break

            target_dirname = os.path.join(root, base_dirname)

            if not os.path.isdir(target_dirname):
                os.makedirs(target_dirname)

            for fn in files:
                src = os.path.join(dirname, fn)
                dst = os.path.join(target_dirname, fn)
                shutil.copy2(src, dst)

        if self.godot_project.set_development_path:
            return

        def copy_python_lib_file(src, dst, fn, ratio, force=False):
            check_existance = True

            if dst.endswith('.py'):
                dst += 'c'
                log.info('compiling %s -> %s' % (src, dst))
                changed = force or not os.path.exists(dst) or os.stat(src).st_mtime != os.stat(dst).st_mtime
                if not changed:
                    return

                python_exe = self.python_dependencies['executable']
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
                log.info('copying %s -> %s' % (src, dst))
                changed = force or not os.path.exists(dst) or os.stat(src).st_mtime != os.stat(dst).st_mtime
                if not changed:
                    return

                target_dir = os.path.dirname(dst)
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)

                shutil.copy2(src, dst)

            if check_existance:
                assert os.path.exists(dst), dst

        log.info('byte-compiling Python sources')
        to_copy = []
        for dirname, subdirs, files in os.walk(self.godot_project.python_package):
            if '__pycache__' in dirname:
                continue
            target_dirname = os.path.join(lib_root, dirname)

            for fn in files:
                src = os.path.join(dirname, fn)
                dst = os.path.join(target_dirname, fn)
                to_copy.append((src, dst, fn))

        for i, (src, dst, fn) in enumerate(to_copy):
            copy_python_lib_file(src, dst, fn, (i+1)/len(to_copy))

    def collect_dependencies(self):
        ziplib = glob.glob(os.path.join(tools_root, '..', '_godopy.cp*.*')).pop()

        if not self.dry_run and not os.path.isdir(os.path.join('build', 'lib')):
            shutil.unpack_archive(ziplib, 'build', 'xztar')

        self.python_dependencies['bin_dir'] = bin_dir = os.path.join('build', 'bin')
        self.python_dependencies['lib_dir'] = os.path.join('build', 'lib')

        self.python_dependencies['py_files'] = []
        self.python_dependencies['zip_dirs'] = set()

        python_exe = 'python.exe'
        self.python_dependencies['executable'] = os.path.join(bin_dir, python_exe)

    def collect_godot_project_data(self, ext):
        self.godot_project = ext

    def collect_godot_generic_library_data(self, ext):
        if self.godot_project is None:
            sys.stderr.write("Can't build a GDNative library without a Godot project.\n\n")
            sys.exit(1)

        if self.gdnative_library_path is not None:
            sys.stderr.write("Can't build multiple GDNative libraries.\n")
            sys.exit(1)

        platform = get_platform()

        ext_name = self.godot_project.get_setuptools_name(ext.name, validate='.gdnlib')
        ext_path = self.get_ext_fullpath(ext_name)
        godot_root = self.godot_root = ensure_godot_project_path(ext_path)

        dst_dir, dst_fullname = os.path.split(ext_path)
        dst_name_parts = dst_fullname.split('.')
        # src_zip = '.'.join(['_godopy', *dst_name_parts[1:]])

        python_exe = self.python_dependencies['executable']
        cmd = [python_exe, '-c', "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True, universal_newlines=True)
        ext_suffix = result.stdout.strip()

        src_name = dst_name = '_godopy' + ext_suffix

        binext_path = os.path.join(godot_root, self.godot_project.binary_path, 'lib', dst_name)

        # self.build_context['godopy_zip_name'] = src_zip
        self.build_context['godopy_library_name'] = src_name
        self.build_context['target'] = binext_path
        self.build_context['library_name'] = '_godopy'

        base_name = dst_name_parts[0]
        gdnlib_respath = make_resource_path(godot_root, os.path.join(dst_dir, base_name + '.gdnlib'))
        setup_script_respath = make_resource_path(godot_root, os.path.join(dst_dir, base_name + '__setup.gd'))
        self.gdnative_library_path = gdnlib_respath
        self.generic_setup = True

        context = dict(singleton=False, load_once=True, reloadable=False)
        context.update(ext._gdnative_options)
        context['symbol_prefix'] = 'godopy_'
        context['libraries'] = {platform: make_resource_path(godot_root, binext_path), 'Server.64': make_resource_path(godot_root, binext_path)}

        # context['main_zip_resource'] = main_zip_res = 'res://%s/%s.pak' % (self.godot_project.binary_path, base_name)
        # context['venv_path'] = self.python_dependencies['site_dir'].replace(os.sep, '/')
        context['lib_path'] = 'res://%s/lib' % self.godot_project.binary_path
        context['libtools_path'] = 'res://%s/libtools' % self.godot_project.binary_path
        if self.godot_project.set_development_path:
            context['development_path'] = self.godot_project.development_path or os.path.realpath(root_dir)
            context['development_path'] = str(context['development_path']).replace(os.sep, '/')

        deps = []

        context['dependencies'] = {platform: deps, 'Server.64': deps}
        context['library'] = 'res://%s' % gdnlib_respath
        context['python_package'] = self.godot_project.python_package

        context['settings'] = create_settings(self.godot_project)

        self.make_godot_resource('gdnlib.mako', gdnlib_respath, context)
        self.make_godot_resource('library_setup.gd.mako', setup_script_respath, context)

    def collect_godot_library_data(self, ext):
        if self.godot_project is None:
            raise SystemExit("Can't build a GDNative library without a Godot project.\n\n")

        if self.gdnative_library_path is not None:
            raise SystemExit("Can't build multiple GDNative libraries.\n")

        platform = get_platform()

        pyx_source = '__init__.pyx'
        py_source = '__init__.py'

        if os.path.exists(os.path.join(self.godot_project.python_package, pyx_source)):
            library_source = pyx_source
        elif os.path.exists(os.path.join(self.godot_project.python_package, py_source)):
            library_source = py_source
        else:
            raise SystemExit("Coudn't find GDNative library source, tried %r and %r" % (pyx_source, py_source))

        ext_name = self.godot_project.get_setuptools_name(ext.name, validate='.gdnlib')
        ext_path = self.get_ext_fullpath(ext_name)
        godot_root = self.godot_root = ensure_godot_project_path(ext_path)

        dst_dir, dst_fullname = os.path.split(ext_path)
        dst_name_parts = dst_fullname.split('.')
        base_name = dst_name_parts[0]
        dst_name_parts[0] = 'lib' + dst_name_parts[0]
        dst_fullname = dst_name_parts[0] + get_dylib_ext()
        staticlib_name = 'libgodopy.%s.%s.64' % (platform_suffix(platform), build_target)

        binext_path = os.path.join(godot_root, self.godot_project.binary_path, platform_suffix(platform), dst_fullname)

        cpp_library_dir = os.path.dirname(library_source)
        self.build_context['cpp_library_path'] = \
            os.path.join(self.godot_project.python_package, cpp_library_dir, '__gdinit__.cpp')

        self.build_context['godopy_library_name'] = staticlib_name
        self.build_context['target'] = make_relative_path(binext_path)
        self.build_context['singleton'] = ext._gdnative_options.get('singleton', False)
        self.build_context['gdnative_options'] = ext._gdnative_options.get('gdnative_options', False)

        dst_name = dst_name_parts[0]
        self.build_context['library_name'] = dst_name
        gdnlib_respath = make_resource_path(godot_root, os.path.join(dst_dir, base_name + '.gdnlib'))
        setup_script_respath = make_resource_path(godot_root, os.path.join(dst_dir, base_name + '__setup.gd'))
        self.gdnative_library_path = gdnlib_respath
        self.generic_setup = False

        context = dict(singleton=False, load_once=True, reloadable=False)
        context.update(ext._gdnative_options)
        context['symbol_prefix'] = 'godopy_'
        context['libraries'] = {platform: make_resource_path(godot_root, binext_path), 'Server.64': make_resource_path(godot_root, binext_path)}

        # context['main_zip_resource'] = main_zip_res = 'res://%s/%s.pak' % (self.godot_project.binary_path, base_name)
        # context['venv_path'] = self.python_dependencies['site_dir'].replace(os.sep, '/')
        context['lib_path'] = 'res://%s/lib' % self.godot_project.binary_path
        context['libtools_path'] = 'res://%s/libtools' % self.godot_project.binary_path
        if self.godot_project.set_development_path:
            context['development_path'] = self.godot_project.development_path or os.path.realpath(root_dir)
            context['development_path'] = str(context['development_path']).replace(os.sep, '/')

        deps = []

        context['dependencies'] = {platform: deps, 'Server.64': deps}
        context['library'] = 'res://%s' % gdnlib_respath
        context['python_package'] = self.godot_project.python_package

        context['settings'] = create_settings(self.godot_project)

        self.make_godot_resource('gdnlib.mako', gdnlib_respath, context)
        self.make_godot_resource('library_setup.gd.mako', setup_script_respath, context)
        self.collect_sources(ext.sources + [library_source])

    def collect_godot_nativescript_data(self, ext):
        if not self.gdnative_library_path:
            raise SystemExit("Can't build a NativeScript extension without a GDNative library.")

        addon = None

        if ext._addon_prefix:
            try:
                addon = self.addons[ext._addon_prefix]
            except KeyError:
                raise SystemExit("Addon %r was not found" % ext._addon_prefix)

        ext_name = self.godot_project.get_setuptools_name(ext.name, addon and addon.name, validate='.gdns')
        ext_path = self.get_ext_fullpath(ext_name)
        godot_root = ensure_godot_project_path(ext_path)

        dst_dir, dst_fullname = os.path.split(ext_path)
        dst_name_parts = dst_fullname.split('.')
        dst_name = dst_name_parts[0]
        gdns_path = os.path.join(dst_dir, dst_name + '.gdns')

        classname = ext._nativescript_classname or dst_name
        context = dict(gdnlib_resource=self.gdnative_library_path, classname=classname)

        self.make_godot_resource('gdns.mako', make_resource_path(godot_root, gdns_path), context)
        self.collect_sources(ext.sources, addon, prepend=True)

        if not self.godot_project.set_development_path:
            self.collect_dynamic_sources(ext._dynamic_sources, addon)

    def collect_godot_addon_data(self, ext):
        self.addons[ext.name] = ext
        config_path = os.path.join('addons', ext.name, 'plugin.cfg')

        context = {
            **ext._addon_metadata,
            'name': ext._addon_metadata.get('name', ext.name),
            'script': ext._addon_metadata.get('script', ext.name + '.gdns'),
        }

        for pattern in ext._data_file_patterns:
            for fn in glob.glob(os.path.join(ext.name, pattern)):
                self.make_godot_file_resource(fn, os.path.join('addons', fn))

        self.make_godot_resource('plugin.cfg.mako', config_path, context)

    def make_godot_resource(self, template_filename, path, context):
        template = Template(filename=os.path.join(templates_dir, template_filename))
        self.godot_resources[path] = template.render(**context)

    def make_godot_file_resource(self, src_path, path):
        self.godot_resource_files[path] = src_path

    def collect_sources(self, sources, addon=None, prepend=False):
        cpp_sources = []
        pyx_sources = []

        for pyx_source in sources:
            if pyx_source.endswith('.cpp'):
                cpp_sources.append(pyx_source)
                continue

            source = os.path.join(self.godot_project.python_package, pyx_source)
            target = source.replace('.pyx', '.cpp')
            target_dir, target_name = os.path.split(target)

            name, _ = os.path.splitext(target_name)

            dir_components = target_dir.split(os.sep)

            if is_internal_path(dir_components[0]):
                dir_components = dir_components[1:]

            if name == '__init__':
                modname = '.'.join(dir_components)
            else:
                modname = '.'.join([*dir_components, name])

            pyx_sources.append({
                'name': modname,
                'symbol_name': modname.replace('.', '__'),
                'cpp': target,
                'pyx': source
            })

        for cpp_source in cpp_sources:
            source = os.path.join(self.godot_project.python_package, cpp_source)
            self.build_context['cpp_sources'].append(make_relative_path(source))

        if prepend:
            self.build_context['pyx_sources'] = pyx_sources + self.build_context['pyx_sources']
        else:
            self.build_context['pyx_sources'] += pyx_sources

    def collect_dynamic_sources(self, sources, addon):
        prefix = addon and addon.name or self.godot_project.python_package

        collection = self.python_dependencies['py_files']

        if sources:
            dirname = os.path.dirname(sources[0])
            if dirname:
                collection.append(('local', os.path.join(prefix, '__init__.py')))
                prefix = os.path.join(prefix, dirname)

        append_init_file = False

        # if addon and addon._editor_only:
        #     collection = self.python_dependencies['py_files_dev']
        #     if prefix not in self.python_dependencies['zip_dirs_dev']:
        #         self.python_dependencies['zip_dirs_dev'].add(prefix)
        #         append_init_file = True
        # else:

        if prefix not in self.python_dependencies['zip_dirs']:
            self.python_dependencies['zip_dirs'].add(prefix)
            append_init_file = True

        if append_init_file:
            init_file = os.path.join(prefix, '__init__.py')

            if os.path.exists(init_file):
                collection.append(('local', init_file))
            else:
                collection.append((None, init_file))

        for source in sources:
            collection.append(('local', os.path.join(prefix, os.path.basename(source))))


def ensure_godot_project_path(path):
    godot_root = detect_godot_project(path)

    if not godot_root:
        sys.stderr.write("No Godot project detected.\n")
        sys.exit(1)

    return os.path.realpath(godot_root)


def detect_godot_project(dir, fn='project.godot'):
    if not dir or not fn:
        return

    if os.path.isdir(dir) and 'project.godot' in os.listdir(dir):
        return dir

    return detect_godot_project(*os.path.split(dir))


def make_resource_path(godot_root, path):
    # Filenames should be exportable to case-sesitive platforms, don't touch them here
    folder, fn = os.path.split(path)
    folder = os.path.realpath(folder).replace(os.path.realpath(godot_root), '').lstrip(os.sep)
    return os.path.join(folder, fn).replace(os.sep, '/')


def make_relative_path(path):
    return os.path.realpath(path).replace(root_dir, '').lstrip(os.sep).replace(os.sep, '/')


def platform_suffix(platform):
    replacements = {'x11': 'linux'}
    suffix = platform.lower().split('.')[0]

    if suffix in replacements:
        return replacements[suffix]

    return suffix


def is_python_source(fn):
    return fn.endswith('.py')


def is_python_ext(fn):
    return fn.endswith('.so') or fn.endswith('.pyd') or fn.endswith('.dll') or fn.endswith('.dylib')


def is_generic_dylib(fn):
    return os.sep + '.' in fn


def get_dylib_ext():
    if sys.platform == 'darwin':
        return '.dylib'
    elif sys.platform == 'win32':
        return '.dll'
    return '.so'


def inner_so_path(root, fn):
    parts = fn.split('.')
    if root == 'site':
        if is_generic_dylib(fn):
            return '_' + fn
        return '_%s.%s' % (parts[0], parts[-1])
    return '%s.%s' % (parts[0], parts[-1])


def get_platform():
    if sys.maxsize <= 2**32:
        raise SystemExit("32-bit platforms are not supported")

    if sys.platform == 'darwin':
        return 'OSX.64'
    elif sys.platform.startswith('linux'):
        return 'X11.64'
    elif sys.platform == 'win32':
        return 'Windows.64'

    raise SystemExit("Can't build for '%s' platform yet" % sys.platform)


def create_settings(project):
    mod = project.module

    settings = {}

    if hasattr(mod, 'NAME'):
        settings['application/config/name'] = mod.NAME

    if hasattr(mod, 'MAIN_SCENE'):
        settings['application/run/main_scene'] = mod.MAIN_SCENE

    if hasattr(mod, 'ICON'):
        settings['application/config/icon'] = mod.ICON

    if hasattr(mod, 'WINDOW_SIZE'):
        width, height = mod.WINDOW_SIZE
        settings['display/window/size/width'] = width
        settings['display/window/size/height'] = height

    # TODO: [input] block

    if hasattr(mod, 'DEFAULT_ENVIRONMENT'):
        settings['rendering/environment/default_environment'] = mod.DEFAULT_ENVIRONMENT

    return settings
