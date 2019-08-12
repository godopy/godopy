import os
import re
import sys
import math
import enum
import shutil
import hashlib
import zipfile
import subprocess

from setuptools import Extension
from setuptools.command.build_ext import build_ext

from mako.template import Template

from ..version import get_version
from ..utils import get_godot_executable


class ExtType(enum.Enum):
    PROJECT = enum.auto()
    GENERIC_LIBRARY = enum.auto()
    LIBRARY = enum.auto()
    NATIVESCRIPT = enum.auto()


root_dir = os.getcwd()  # XXX
tools_root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
templates_dir = os.path.join(tools_root, 'setup', 'templates')


class GenericGDNativeLibrary(Extension):
    def __init__(self, name):
        self._gdnative_type = ExtType.GENERIC_LIBRARY
        super().__init__(name, sources=[])


class GDNativeLibrary(Extension):
    def __init__(self, name, source, extra_sources=None, **gdnative_options):
        self._gdnative_type = ExtType.LIBRARY
        self._gdnative_options = gdnative_options

        sources = [source]

        if extra_sources is not None:
            for src in extra_sources:
                sources.append(src)

        super().__init__(name, sources=sources)


class NativeScript(Extension):
    def __init__(self, name, *, sources=None, class_name=None):
        self._gdnative_type = ExtType.NATIVESCRIPT
        self._nativescript_classname = class_name
        super().__init__(name, sources=(sources or []))


class gdnative_build_ext(build_ext):
    godot_project = None
    gdnative_library_path = None
    generic_setup = False

    build_context = {
        '__version__': get_version(),
        'godot_headers_path': os.path.normpath(os.path.join(tools_root, '..', 'godot_headers')),
        'pygodot_bindings_path': os.path.dirname(tools_root),
        'singleton': False,
        'pyx_sources': [],
        'cpp_sources': []
    }

    godot_resources = {}
    python_dependencies = {}

    def run(self):
        if 'VIRTUAL_ENV' not in os.environ:
            sys.stderr.write("Please run this command inside the virtual environment.\n")
            sys.exit(1)

        dependencies_collected = False

        for ext in self.extensions:
            if self.godot_project:
                if not dependencies_collected:
                    print('collecting Python dependencies')
                    self.collect_dependencies()
                    dependencies_collected = True

                print('setting up', 'GDNative {0}: {1}'.format(ext._gdnative_type.name.lower(), ext.name))

            getattr(self, 'collect_godot_{0}_data'.format(ext._gdnative_type.name.lower()))(ext)

        if self.generic_setup:
            self.run_copylib()
        else:
            self.run_build()

        for res_path, content in self.godot_resources.items():
            self.write_target_file(os.path.join(self.godot_project.name, res_path), content,
                                   pretty_path='res://%s' % res_path, is_editable_resource=True)

        self.package_dependencies()

        setup_script = self.gdnative_library_path.replace('.gdnlib', '__setup.gd')
        print('running "res://%s"' % setup_script)
        if not self.dry_run:
            subprocess.run([
                get_godot_executable(),
                '--path', self.godot_project.name,
                '-s', 'res://%s' % setup_script
            ], check=True)

    def run_copylib(self):
        source = os.path.join(self.build_context['pygodot_bindings_path'], self.build_context['pygodot_library_name'])
        target = self.build_context['target']

        print(source)
        assert os.path.exists(source)
        print(target, os.path.exists(target), os.path.islink(target))

        # TODO: Copy, don't link
        if os.path.islink(target) and os.readlink(target) == source and not self.force:
            print('skip linking "%s"' % make_relative_path(target))
            return

        # Remove if the target is not a correct symlink
        if (os.path.exists(target) or os.path.islink(target)) and not self.dry_run:
            os.unlink(target)

        target_dir = os.path.dirname(target)
        if not os.path.exists(target_dir) and not self.dry_run:
            os.makedirs(target_dir)

        print('linking "%s"' % make_relative_path(target))
        if not self.dry_run:
            os.symlink(source, target)

    def run_build(self):
        cpp_lib_template = Template(filename=os.path.join(templates_dir, 'gdlibrary.cpp.mako'))
        cpp_lib_path = self.build_context['cpp_library_path']
        self.write_target_file(cpp_lib_path, cpp_lib_template.render(**self.build_context))

        self.build_context['cpp_sources'].append(make_relative_path(self.build_context['cpp_library_path']))

        scons_template = Template(filename=os.path.join(templates_dir, 'SConstruct.mako'))
        self.write_target_file('SConstruct', scons_template.render(**self.build_context))

        scons = os.path.join(sys.prefix, 'Scripts', 'scons') if sys.platform == 'win32' else 'scons'

        if not self.dry_run:
            self.spawn([scons])

    def write_target_file(self, path, content, pretty_path=None, is_editable_resource=False):
        if pretty_path is None:
            pretty_path = path

        if os.path.exists(path) and not self.force:
            # Check if there are any changes
            new_hash = hashlib.sha1(content.encode('utf-8'))
            with open(path, encoding='utf-8') as fp:
                old_hash = hashlib.sha1(fp.read().encode('utf-8'))
            if old_hash.digest() == new_hash.digest():
                print('skip writing "%s"' % pretty_path)
                return
            elif is_editable_resource and not self.force:
                # Do not overwrite user resources without --force flag
                print('WARNING! modified resource already exists: "%s"' % pretty_path)
                return

        print('writing "%s"' % pretty_path, path)
        if not self.dry_run:
            dirname, basename = os.path.split(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(path, 'w', encoding='utf-8') as fp:
                fp.write(content)

    def package_dependencies(self):
        binroot = os.path.join(self.godot_project.name, self.godot_project.binary_path, platform_suffix(get_platform()))

        for d in sorted(self.python_dependencies['bin_dirs']):
            target_dir = os.path.join(binroot, d)
            if not os.path.isdir(target_dir) and not self.dry_run:
                os.makedirs(target_dir)

        for root, fn in reversed(self.python_dependencies['so_files'] + self.python_dependencies['so_files_ed']):
            if root == 'dynload':
                basedir = self.python_dependencies['dynload_dir']
            elif root == 'site':
                basedir = self.python_dependencies['site_dir']
            else:
                basedir = self.python_dependencies['mainlib_dir']

            src = os.path.join(basedir, fn)
            dst = os.path.join(binroot, fn)
            if not os.path.exists(dst) and not self.dry_run:
                shutil.copy2(src, dst)

        _, gdnlib_name = os.path.split(self.gdnative_library_path)
        basename, _ = os.path.splitext(gdnlib_name)
        main_zip_path = os.path.join(self.godot_project.name, self.godot_project.binary_path, '%s.pak' % basename)
        tools_zip_path = os.path.join(self.godot_project.name, self.godot_project.binary_path, '%s-dev.pak' % basename)

        self._make_zip(main_zip_path, 'py_files')
        self._make_zip(tools_zip_path, 'py_files_ed')

    def _make_zip(self, zippath, files):
        print('byte-compiling and compressing Python dependencies into "res://%s"' % zippath)
        if self.dry_run:
            return

        builddir_src = os.path.join('build', os.path.basename(zippath) + '.py_files')
        builddir_dst = os.path.join('build', os.path.basename(zippath) + '.pyc_files')

        verbosity = 1

        for d in sorted(self.python_dependencies['zip_dirs']):
            for target_dir in (os.path.join(builddir_src, d), os.path.join(builddir_dst, d)):
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)

        _prev_ratio = 0
        _total = len(self.python_dependencies[files])
        with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as _zip:
            for i, (root, fn) in enumerate(self.python_dependencies[files]):
                if root == 'lib':
                    basedir = self.python_dependencies['lib_dir']
                elif root == 'site':
                    basedir = self.python_dependencies['site_dir']
                else:
                    basedir = None

                pre_dst = os.path.join(builddir_src, fn)

                if basedir is None:
                    with open(pre_dst, 'w', encoding='utf-8'):
                        pass
                else:
                    src = os.path.join(basedir, fn)
                    changed = not os.path.exists(pre_dst) or os.stat(src).st_mtime != os.stat(pre_dst).st_mtime
                    if changed:
                        shutil.copy2(src, pre_dst)

                fnc = fn + 'c'
                dst = os.path.join(builddir_dst, fnc)

                _ratio = math.floor(i * 78 / _total)

                changed = not os.path.exists(dst) or os.stat(pre_dst).st_mtime != os.stat(dst).st_mtime

                if changed:
                    # Compile to bytecode with target interpreter
                    if verbosity > 1:
                        print("Byte-compiling and compressing %r" % fnc)
                    elif verbosity == 1:
                        if _ratio > _prev_ratio:
                            print('.', end='', flush=True)
                            _prev_ratio = _ratio

                    subprocess.run([
                        self.python_dependencies['executable'],
                        '-c',
                        "from py_compile import compile; compile(%r, %r, dfile=%r)" % (pre_dst, dst, fn)
                    ], check=True)
                    shutil.copystat(pre_dst, dst)
                else:
                    if verbosity > 1:
                        print("Compressing %r" % fnc)
                    elif verbosity == 1:
                        if _ratio > _prev_ratio:
                            print('.', end='', flush=True)
                            _prev_ratio = _ratio

                with open(dst, 'rb') as fp_src:
                    with _zip.open(fnc, 'w') as fp_dst:
                        fp_dst.write(fp_src.read())

        if verbosity == 1:
            print()

    def collect_dependencies(self):
        self.python_dependencies['bin_dir'] = bin_dir = os.path.normpath(os.path.join(tools_root, '..', 'buildenv', 'bin'))
        self.python_dependencies['mainlib_dir'] = mainlib_dir = os.path.normpath(os.path.join(tools_root, '..', 'buildenv', 'lib'))
        self.python_dependencies['lib_dir'] = lib_dir = os.path.join(mainlib_dir, 'python3.8')
        self.python_dependencies['dynload_dir'] = dynload_dir = os.path.join(lib_dir, 'lib-dynload')
        self.python_dependencies['site_dir'] = site_dir = os.path.join(lib_dir, 'site-packages')

        self.python_dependencies['py_files'] = py_files = []
        self.python_dependencies['so_files'] = so_files = []
        self.python_dependencies['py_files_ed'] = py_files_ed = []
        self.python_dependencies['so_files_ed'] = so_files_ed = []

        self.python_dependencies['zip_dirs'] = dirs = set()
        self.python_dependencies['bin_dirs'] = so_dirs = set()

        # TODO: non-debug targets
        mainlib = 'libpython3.8d.so'
        python_exe = 'python3.8'
        if sys.platform == 'darwin':
            mainlib = 'libpython3.8d.dylib'
        elif sys.platform == 'win32':
            mainlib = 'python38d.dll'
            python_exe = 'python38.exe'

        self.python_dependencies['executable'] = os.path.join(bin_dir, python_exe)

        so_files.append(('mainlib', mainlib))

        py_files.append((None, os.path.join('godot', '__init__.py')))
        dirs.add('godot')

        py_files.append((None, os.path.join(self.godot_project.shadow_name, '__init__.py')))
        dirs.add(self.godot_project.shadow_name)

        for dirpath, dirnames, filenames in os.walk(lib_dir):
            dirpath = dirpath[len(lib_dir):].lstrip(os.sep)
            skip = False
            if '__pychache__' in dirnames:
                dirnames.remove('__pychache__')
            if '__pycache__' in dirpath:
                continue

            for skipdir in ('site-packages', 'lib-dynload', 'config-', 'lib2to3', 'tkinter', 'ensurepip', 'venv', 'parser', 'test'):
                if dirpath.startswith(skipdir):
                    skip = True
                    break
            if skip:
                continue
            has_files = False
            has_ed_files = False
            for fn in filenames:
                if not is_python_source(fn):
                    continue
                is_tool = dirpath.endswith('tests')
                if not is_tool:
                    for tooldir in ('typing', 'pydoc', 'doctest', 'unittest', 'idlelib', 'distutils', 'zipapp'):
                        if dirpath.startswith(tooldir):
                            is_tool = True
                            break
                if is_tool:
                    has_ed_files = True
                    py_files_ed.append(('lib', os.path.join(dirpath, fn)))
                else:
                    has_files = True
                    py_files.append(('lib', os.path.join(dirpath, fn)))

            if has_files or has_ed_files:
                dirs.add(dirpath)

        for fn in os.listdir(dynload_dir):
            if not is_python_ext(fn):
                continue
            if 'test' in fn:
                so_files_ed.append(('dynload', fn))
            else:
                so_files.append(('dynload', fn))

        for dirpath, dirnames, filenames in os.walk(site_dir):
            dirpath = dirpath[len(site_dir):].lstrip(os.sep)
            skip = False
            if '__pychache__' in dirnames:
                dirnames.remove('__pychache__')
            if '__pycache__' in dirpath:
                continue
            if 'typeshed' in dirpath:
                continue

            for skipdir in ('pip', 'setuptools', 'pkg_resources'):
                if dirpath.startswith(skipdir):
                    skip = True
                    break
            if skip:
                continue
            has_files = False
            has_so_files = False
            for fn in filenames:
                if is_python_source(fn):
                    if dirpath.startswith('traitlets'):
                        continue
                    is_tool = dirpath.endswith('tests') or 'testing' in dirpath
                    for tooldir in ('Cython', 'IPython', 'ipython_genutils', 'jedi', 'parso', 'pexpect', 'traitlets', 'ptyprocess'):
                        if dirpath.startswith(tooldir):
                            is_tool = True
                            break
                    has_files = True
                    if is_tool:
                        py_files_ed.append(('site', os.path.join(dirpath, fn)))
                    else:
                        py_files.append(('site', os.path.join(dirpath, fn)))
                elif is_python_ext(fn):
                    has_so_files = True
                    if 'tests' not in fn and '_dummy' not in fn and not dirpath.startswith('Cython'):
                        so_files.append(('site', os.path.join(dirpath, fn)))
                    else:
                        so_files_ed.append(('site', os.path.join(dirpath, fn)))
            if has_files:
                dirs.add(dirpath)
            if has_so_files:
                so_dirs.add(dirpath)

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
        godot_root = ensure_godot_project_path(ext_path)

        dst_dir, dst_fullname = os.path.split(ext_path)
        dst_name_parts = dst_fullname.split('.')
        src_name = '.'.join(['_pygodot', *dst_name_parts[1:]])

        binext_path = os.path.join(godot_root, self.godot_project.binary_path, platform_suffix(platform), dst_fullname)

        self.build_context['pygodot_library_name'] = src_name
        self.build_context['target'] = binext_path
        self.build_context['library_name'] = '_pygodot'

        dst_name = dst_name_parts[0]
        gdnlib_respath = make_resource_path(godot_root, os.path.join(dst_dir, dst_name + '.gdnlib'))
        self.gdnative_library_path = gdnlib_respath
        self.generic_setup = True
        so_files = self.python_dependencies['so_files'] + self.python_dependencies['so_files_ed']
        deps = ['"res://%s/%s/%s"' % (self.godot_project.binary_path, platform_suffix(platform), fn) for root, fn in so_files]
        context = dict(
            singleton=False,
            load_once=True,
            symbol_prefix='pygodot_',
            reloadable=False,
            libraries={platform: make_resource_path(godot_root, binext_path)},
            dependencies={platform: deps}
        )

        self.make_godot_resource('gdnlib.mako', gdnlib_respath, context)

    def collect_godot_library_data(self, ext):
        if self.godot_project is None:
            sys.stderr.write("Can't build a GDNative library without a Godot project.\n\n")
            sys.exit(1)

        if self.gdnative_library_path is not None:
            sys.stderr.write("Can't build multiple GDNative libraries.\n")
            sys.exit(1)

        platform = get_platform()

        ext_name = self.godot_project.get_setuptools_name(ext.name, validate='.gdnlib')
        ext_path = self.get_ext_fullpath(ext_name)
        godot_root = ensure_godot_project_path(ext_path)

        dst_dir, dst_fullname = os.path.split(ext_path)
        dst_name_parts = dst_fullname.split('.')
        base_name = dst_name_parts[0]
        dst_name_parts[0] = 'lib' + dst_name_parts[0]
        # if dst_name_parts[0] == 'gdlibrary':
        #    dst_name_parts[0] = '_gdlibrary'
        # _ext = dst_name_parts[-1]
        dst_fullname = dst_name_parts[0] + get_dylib_ext()
        # dst_fullname = '.'.join(dst_name_parts)
        staticlib_name = 'libpygodot.%s.debug.64' % platform_suffix(platform)

        binext_path = os.path.join(godot_root, self.godot_project.binary_path, platform_suffix(platform), dst_fullname)

        cpp_library_base_path = re.sub(r'\.pyx?$', '.cpp', ext.sources[0])
        cpp_library_dir, cpp_library_unprefixed = os.path.split(cpp_library_base_path)
        self.build_context['cpp_library_path'] = \
            os.path.join(self.godot_project.shadow_name, cpp_library_dir, '_' + cpp_library_unprefixed)

        self.build_context['pygodot_library_name'] = staticlib_name
        self.build_context['target'] = make_relative_path(binext_path)

        dst_name = dst_name_parts[0]
        self.build_context['library_name'] = dst_name
        gdnlib_respath = make_resource_path(godot_root, os.path.join(dst_dir, base_name + '.gdnlib'))
        setup_script_respath = make_resource_path(godot_root, os.path.join(dst_dir, base_name + '__setup.gd'))
        self.gdnative_library_path = gdnlib_respath
        self.generic_setup = False

        context = dict(singleton=False, load_once=True, reloadable=False)
        context.update(ext._gdnative_options)
        context['symbol_prefix'] = 'pygodot_'
        context['libraries'] = {platform: make_resource_path(godot_root, binext_path), 'Server.64': make_resource_path(godot_root, binext_path)}
        so_files = self.python_dependencies['so_files'] + self.python_dependencies['so_files_ed']
        context['main_zip_resource'] = main_zip_res = 'res://%s/%s.pak' % (self.godot_project.binary_path, base_name)
        context['dev_zip_resource'] = tools_zip_res = 'res://%s/%s-dev.pak' % (self.godot_project.binary_path, base_name)
        deps = [main_zip_res, tools_zip_res, *('res://%s/%s/%s' % (self.godot_project.binary_path, platform_suffix(platform), fn) for root, fn in so_files)]
        context['dependencies'] = {platform: deps, 'Server.64': deps}
        context['library'] = 'res://%s' % gdnlib_respath

        self.make_godot_resource('gdnlib.mako', gdnlib_respath, context)
        self.make_godot_resource('library_setup.gd.mako', setup_script_respath, context)
        self.collect_sources(ext.sources)

    def collect_godot_nativescript_data(self, ext):
        if not self.gdnative_library_path:
            sys.stderr.write("Can't build a NativeScript extension without a GDNative library.\n")
            sys.exit(1)

        ext_name = self.godot_project.get_setuptools_name(ext.name, validate='.gdns')
        ext_path = self.get_ext_fullpath(ext_name)
        godot_root = ensure_godot_project_path(ext_path)

        dst_dir, dst_fullname = os.path.split(ext_path)
        dst_name_parts = dst_fullname.split('.')
        dst_name = dst_name_parts[0]
        gdns_path = os.path.join(dst_dir, dst_name + '.gdns')

        # if dst_name == self.build_context['library_name']:
        #     raise NameError("'%s' name is already used. Please select a different name\n" % dst_name)

        classname = ext._nativescript_classname or dst_name
        context = dict(gdnlib_resource=self.gdnative_library_path, classname=classname)

        self.make_godot_resource('gdns.mako', make_resource_path(godot_root, gdns_path), context)
        self.collect_sources(ext.sources)

    def make_godot_resource(self, template_filename, path, context):
        template = Template(filename=os.path.join(templates_dir, template_filename))
        self.godot_resources[path] = template.render(**context)

    def collect_sources(self, sources):
        cpp_sources = []

        for pyx_source in sources:
            if pyx_source.endswith('.cpp'):
                cpp_sources.append(pyx_source)
                continue

            source = os.path.join(self.godot_project.shadow_name, pyx_source)
            target = source.replace('.pyx', '.cpp')
            target_dir, target_name = os.path.split(target)

            # varnames should be unique due to the way Python modules are initialized
            varname, _ = os.path.splitext(target_name)

            self.build_context['pyx_sources'].append((varname, target, source))

        for cpp_source in cpp_sources:
            source = os.path.join(self.godot_project.shadow_name, cpp_source)
            self.build_context['cpp_sources'].append(make_relative_path(source))


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
    return path.replace(godot_root, '').lstrip(os.sep).replace(os.sep, '/')


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
    return fn.endswith('.so') or fn.endswith('.pyd')


def get_dylib_ext():
    if sys.platform == 'darwin':
        return '.dylib'
    elif sys.platform == 'win32':
        return '.dll'
    return '.so'


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
