import os
import sys
import math
import glob
import shutil
import hashlib
import zipfile
import subprocess

from setuptools import Extension
from setuptools.command.build_ext import build_ext

from mako.template import Template

from ..version import get_version
from ..utils import get_godot_executable, is_internal_path

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

TOOL_PACKAGES = ('Cython', 'IPython', 'ipython_genutils', 'jedi', 'parso', 'pexpect', 'traitlets', 'ptyprocess')


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
            self.write_target_file(os.path.join(self.godot_project.path_prefix, res_path), content,
                                   pretty_path='res://%s' % res_path, is_editable_resource=True)

        for res_path, source in self.godot_resource_files.items():
            target = os.path.join(self.godot_project.path_prefix, res_path)
            if os.path.exists(target) and os.stat(source).st_mtime == os.stat(target).st_mtime and not self.force:
                print('skip copying "%s"' % make_relative_path(target))
                continue

            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir) and not self.dry_run:
                os.makedirs(target_dir)

            print('copying "%s"' % make_relative_path(target))
            if not self.dry_run:
                shutil.copy2(source, target)

        self.package_dependencies()

        setup_script = self.gdnative_library_path.replace('.gdnlib', '__setup.gd')
        print('updating Godot project settings, running "res://%s"' % setup_script)
        if not self.dry_run:
            subprocess.run([
                get_godot_executable(),
                '--path', self.godot_project.path_prefix,
                '-s', 'res://%s' % setup_script
            ], check=True)

        print('removing "res://%s"' % setup_script)
        if not self.dry_run:
            os.unlink(os.path.join(self.godot_project.path_prefix, setup_script))

    def run_copylib(self):
        source = os.path.join(self.build_context['godopy_bindings_path'], self.build_context['godopy_library_name'])
        target = self.build_context['target']

        # print(source)
        assert os.path.exists(source)

        # print(target, os.path.exists(target), os.path.islink(target))

        if os.path.exists(target) and os.stat(source).st_mtime == os.stat(target).st_mtime and not self.force:
            print('skip copying "%s"' % make_relative_path(target))
            return

        target_dir = os.path.dirname(target)
        if not os.path.exists(target_dir) and not self.dry_run:
            os.makedirs(target_dir)

        print('copying "%s"' % make_relative_path(target))
        if not self.dry_run:
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
        binroot = os.path.join(self.godot_project.path_prefix, self.godot_project.binary_path, platform_suffix(get_platform()))

        for d in sorted(self.python_dependencies['bin_dirs']):
            target_dir = os.path.join(binroot, d)
            if not os.path.isdir(target_dir) and not self.dry_run:
                os.makedirs(target_dir)

        for root, fn in reversed(self.python_dependencies['so_files']):
            if root == 'dynload':
                basedir = self.python_dependencies['dynload_dir']
            elif root == 'site':
                basedir = self.python_dependencies['site_dir']
            else:
                basedir = self.python_dependencies['mainlib_dir']

            src = os.path.join(basedir, fn)
            dst = os.path.join(binroot, inner_so_path(root, fn))
            dstdir = os.path.dirname(dst)
            if not os.path.isdir(dstdir) and not self.dry_run:
                os.makedirs(dstdir)
            if not os.path.exists(dst) and not self.dry_run:
                shutil.copy2(src, dst)

        _, gdnlib_name = os.path.split(self.gdnative_library_path)
        basename, _ = os.path.splitext(gdnlib_name)
        main_zip_path = os.path.join(self.godot_project.path_prefix, self.godot_project.binary_path, '%s.pak' % basename)
        # tools_zip_path = os.path.join(self.godot_project.path_prefix, self.godot_project.binary_path, '%s-dev.pak' % basename)

        self._make_zip(main_zip_path, 'py_files', 'zip_dirs')
        # self._make_zip(tools_zip_path, 'py_files_dev', 'zip_dirs_dev')

        if self.dry_run:
            return

        builddir_src = os.path.join('build', '_bin.py_files')

        for path in self.python_dependencies['py_files_for_bin']:
            d, fn = os.path.split(path)
            assert fn == '__init__.py', fn
            src_dir = os.path.join(builddir_src, d)
            if not os.path.isdir(src_dir):
                os.makedirs(src_dir)
            src = os.path.join(builddir_src, d, fn)
            with open(src, 'w', encoding='utf-8'):
                pass

            dst = os.path.join(binroot, d, fn + 'c')
            cmd = [self.python_dependencies['executable'], '-c',
                   "from py_compile import compile; compile(%r, %r, doraise=True)" % (src, dst)]

            subprocess.run(cmd, check=True)

    def _make_zip(self, zippath, files, dirs):
        print('byte-compiling and compressing Python dependencies into "res://%s"' % zippath)
        if self.dry_run:
            return

        builddir_src = os.path.join('build', os.path.basename(zippath) + '.py_files')
        builddir_dst = os.path.join('build', os.path.basename(zippath) + '.pyc_files')
        unprefixed_binroot = os.path.join(self.godot_project.binary_path, platform_suffix(get_platform()))

        verbosity = 1

        for d in sorted(self.python_dependencies[dirs]):
            for target_dir in (os.path.join(builddir_src, d), os.path.join(builddir_dst, d)):
                if not os.path.isdir(target_dir):
                    os.makedirs(target_dir)

        _prev_ratio = 0
        so_shims = [(root, fn) for root, fn in
                    reversed(self.python_dependencies['so_files'])
                    if root == 'site']
        _total = len(self.python_dependencies[files])

        so_shims_written = set()
        with zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED) as _zip:
            for root, fn_so in so_shims:
                base_fn, so_ext = os.path.splitext(fn_so)
                fn = base_fn + '.py'
                inner_dir, import_name = os.path.split(base_fn)

                if os.path.basename(inner_dir).startswith('.'):
                    # Helper libraries not intended for import
                    # TODO: mv path/to/.bin/osx/_numpy/.dylibs path/to/.bin/osx/
                    continue

                bin_dir = '/' + os.path.join(unprefixed_binroot, inner_dir)
                import_name = import_name.split('.')[0]

                fn = os.path.join(inner_dir, import_name + '.py')
                fnc = fn + 'c'
                dst = os.path.join(builddir_dst, fnc)
                pre_dst = os.path.join(builddir_src, fn)

                pre_dst_dir = os.path.dirname(pre_dst)
                if not os.path.isdir(pre_dst_dir):
                    os.makedirs(pre_dst_dir)

                with open(pre_dst, 'w', encoding='utf-8') as fp:
                    # TODO: Fix NumPy 1.18 initialization and remove hacks
                    shim_tmpl = (
                        "try:\n"
                        "    import _{1} as ___mod\n\n"
                        "    for ___name in dir(___mod):\n"
                        "        globals()[___name] = getattr(___mod, ___name, None)\n"
                        "except Exception as ex:\n"
                        "    # print('Error ignored during \\'{1}\\' extension init: %s' % ex)\n"
                        "    RandomState = None\n"
                        "    Philox = None\n"
                        "    PCG64 = None\n"
                        "    SFC64 = None\n"
                        "    Generator = None\n"
                        "    MT19937 = None\n"
                        "    default_rng = None\n"
                        "    SeedSequence = None\n"
                        "    BitGenerator = None\n"
                    )

                    fp.write(shim_tmpl.format(bin_dir, os.path.join(inner_dir, import_name).replace(os.sep, '.')))

                assert os.path.exists(pre_dst), pre_dst

                cmd = [
                    self.python_dependencies['executable'],
                    '-c',
                    "from py_compile import compile; compile(%r, %r, dfile=%r, doraise=True)" % (pre_dst, dst, fn)
                ]

                subprocess.run(cmd, check=True)
                assert os.path.exists(dst), dst
                shutil.copystat(pre_dst, dst)

                with open(dst, 'rb') as fp_src:
                    with _zip.open(fnc, 'w') as fp_dst:
                        fp_dst.write(fp_src.read())
                so_shims_written.add(fnc)

            for i, (root, fn) in enumerate(self.python_dependencies[files]):
                if root == 'lib':
                    basedir = self.python_dependencies['lib_dir']
                elif root == 'site':
                    basedir = self.python_dependencies['site_dir']
                elif root == 'local':
                    basedir = root_dir
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

                assert os.path.exists(pre_dst), pre_dst

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

                    cmd = [
                        self.python_dependencies['executable'],
                        '-c',
                        "from py_compile import compile; compile(%r, %r, dfile=%r)" % (pre_dst, dst, fn)
                    ]

                    if sys.version_info >= (3, 7):
                        subprocess.run(cmd, check=True, capture_output=True)
                    else:
                        subprocess.run(cmd, check=True)

                    assert os.path.exists(dst), dst

                    shutil.copystat(pre_dst, dst)
                else:
                    if verbosity > 1:
                        print("Compressing %r" % fnc)
                    elif verbosity == 1:
                        if _ratio > _prev_ratio:
                            print('.', end='', flush=True)
                            _prev_ratio = _ratio

                if fnc not in so_shims_written:
                    assert os.path.exists(dst), dst

                    with open(dst, 'rb') as fp_src:
                        with _zip.open(fnc, 'w') as fp_dst:
                            fp_dst.write(fp_src.read())

        if verbosity == 1:
            print()

    def collect_dependencies(self):
        # TODO: NumPy dependencies in .dynlibs

        if sys.platform == 'win32':
            py_base_dir = os.path.normpath(os.path.join(tools_root, '..', 'deps', 'python'))
            py_venv_dir = os.path.normpath(os.path.join(tools_root, '..', 'venv'))
            self.python_dependencies['bin_dir'] = bin_dir = os.path.join(py_base_dir, 'PCBuild', 'amd64')
            self.python_dependencies['mainlib_dir'] = mainlib_dir = bin_dir
            self.python_dependencies['lib_dir'] = lib_dir = os.path.join(py_base_dir, 'Lib')
            self.python_dependencies['dynload_dir'] = dynload_dir = bin_dir
            self.python_dependencies['site_dir'] = site_dir = os.path.join(py_venv_dir, 'Lib', 'site-packages')
        else:
            py_base_dir = os.path.normpath(os.path.join(tools_root, '..', 'deps', 'python'))
            py_venv_dir = os.path.normpath(os.path.join(tools_root, '..', 'venv'))
            self.python_dependencies['bin_dir'] = bin_dir = os.path.join(py_base_dir, 'build', 'bin')
            self.python_dependencies['mainlib_dir'] = mainlib_dir = os.path.join(py_base_dir, 'build', 'lib')
            self.python_dependencies['lib_dir'] = lib_dir = os.path.join(mainlib_dir, 'python3.8')
            self.python_dependencies['dynload_dir'] = dynload_dir = os.path.join(lib_dir, 'lib-dynload')
            self.python_dependencies['site_dir'] = site_dir = os.path.join(py_venv_dir, 'lib', 'python3.8', 'site-packages')

        self.python_dependencies['py_files'] = py_files = []
        self.python_dependencies['so_files'] = so_files = []
        self.python_dependencies['py_files_for_bin'] = py_files_for_bin = []
        # self.python_dependencies['py_files_dev'] = py_files_dev = []
        # self.python_dependencies['so_files_dev'] = so_files_dev = []
        # self.python_dependencies['py_files_for_bin_dev'] = py_files_for_bin_dev = []

        self.python_dependencies['zip_dirs'] = dirs = set()
        # self.python_dependencies['zip_dirs_dev'] = dirs_dev = set()
        self.python_dependencies['bin_dirs'] = so_dirs = set()

        mainlib = None
        extra_mainlib = None
        python_exe = 'python3'
        if sys.platform == 'win32':
            mainlib = 'python38.dll'
            # extra_mainlib = 'python38_d.dll'
            python_exe = 'python.exe'

        self.python_dependencies['executable'] = os.path.join(bin_dir, python_exe)

        if mainlib is not None:
            so_files.append(('mainlib', mainlib))
        if extra_mainlib is not None:
            so_files.append(('mainlib', extra_mainlib))

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
            # has_dev_files = False
            for fn in filenames:
                if not is_python_source(fn):
                    continue
                is_tool = dirpath.endswith('tests')
                if not is_tool:
                    for tooldir in ('typing', 'pydoc', 'doctest', 'idlelib', 'distutils', 'zipapp'):
                        if dirpath.startswith(tooldir):
                            is_tool = True
                            break
                if is_tool:
                    pass
                    # has_dev_files = True
                    # py_files_dev.append(('lib', os.path.join(dirpath, fn)))
                else:
                    has_files = True
                    py_files.append(('lib', os.path.join(dirpath, fn)))

            if has_files:
                dirs.add(dirpath)
            # elif has_dev_files:
            #     dirs_dev.add(dirpath)

        for fn in os.listdir(dynload_dir):
            if not is_python_ext(fn):
                continue
            if 'test' in fn:
                pass
                # so_files_dev.append(('dynload', fn))
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
            # has_dev_files = False
            has_so_files = False
            for fn in filenames:
                if is_python_source(fn):
                    # if dirpath.startswith('traitlets') or dirpath.startswith('jedi'):
                    #     continue
                    is_tool = False  # dirpath.endswith('tests')  # or 'testing' in dirpath
                    for tooldir in TOOL_PACKAGES:
                        if dirpath.startswith(tooldir):
                            is_tool = True
                            break
                    if is_tool:
                        pass
                        # has_dev_files = True
                        # py_files_dev.append(('site', os.path.join(dirpath, fn)))
                    else:
                        has_files = True
                        py_files.append(('site', os.path.join(dirpath, fn)))
                elif is_python_ext(fn):
                    has_so_files = True
                    has_files = True
                    if '_dummy' not in fn and not dirpath.startswith('Cython'):
                        so_files.append(('site', os.path.join(dirpath, fn)))
                    # else:
                    #     so_files_dev.append(('site', os.path.join(dirpath, fn)))
            if has_files:
                dirs.add(dirpath)
            # if has_dev_files:
            #     dirs_dev.add(dirpath)
            if has_so_files:
                so_dirs.add('_' + dirpath)

        bin_package_dirs = {''}
        # bin_package_dev_dirs = {''}
        packages_sets = (
            (bin_package_dirs, so_files),
            # (bin_package_dev_dirs, so_files_dev)
        )
        for packages_set, files in packages_sets:
            for root, fn in reversed(files):
                if root == 'site':
                    prefix = '_'
                    create_package = True
                else:
                    prefix = ''
                    create_package = False

                if create_package:
                    dirs, filename = os.path.split(fn)
                    cur_dir = []
                    for d in (prefix + dirs).split(os.sep):
                        cur_dir.append(d)
                        packages_set.add(os.sep.join(cur_dir))

        for bin_dirs, files in ((bin_package_dirs, py_files_for_bin),):
            for d in bin_dirs:
                files.append(os.path.join(d, '__init__.py'))

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
        src_name = '.'.join(['_godopy', *dst_name_parts[1:]])

        binext_path = os.path.join(godot_root, self.godot_project.binary_path, platform_suffix(platform), dst_fullname)

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

        context['main_zip_resource'] = main_zip_res = 'res://%s/%s.pak' % (self.godot_project.binary_path, base_name)
        # context['dev_zip_resource'] = tools_zip_res = 'res://%s/%s-dev.pak' % (self.godot_project.binary_path, base_name)
        context['venv_path'] = self.python_dependencies['site_dir']
        if self.godot_project.set_development_path:
            context['development_path'] = os.path.realpath(root_dir)

        so_files = self.python_dependencies['so_files']
        deps = [main_zip_res,
                *('res://%s/%s/%s' % (self.godot_project.binary_path, platform_suffix(platform), inner_so_path(root, fn))
                    for root, fn in so_files)]
        py_files_for_bin = self.python_dependencies['py_files_for_bin']
        deps += ['res://%s/%s/%sc' % (self.godot_project.binary_path, platform_suffix(platform), fn) for fn in py_files_for_bin]

        context['dependencies'] = {platform: deps, 'Server.64': deps}
        context['library'] = 'res://%s' % gdnlib_respath
        context['python_package'] = self.godot_project.python_package

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
        godot_root = ensure_godot_project_path(ext_path)

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

        context['main_zip_resource'] = main_zip_res = 'res://%s/%s.pak' % (self.godot_project.binary_path, base_name)
        # context['dev_zip_resource'] = tools_zip_res = 'res://%s/%s-dev.pak' % (self.godot_project.binary_path, base_name)
        context['venv_path'] = self.python_dependencies['site_dir']
        if self.godot_project.set_development_path:
            context['development_path'] = os.path.realpath(root_dir)

        so_files = self.python_dependencies['so_files']
        deps = [main_zip_res,
                *('res://%s/%s/%s' % (self.godot_project.binary_path, platform_suffix(platform), inner_so_path(root, fn))
                    for root, fn in so_files)]
        py_files_for_bin = self.python_dependencies['py_files_for_bin']
        deps += ['res://%s/%s/%sc' % (self.godot_project.binary_path, platform_suffix(platform), fn) for fn in py_files_for_bin]

        context['dependencies'] = {platform: deps, 'Server.64': deps}
        context['library'] = 'res://%s' % gdnlib_respath
        context['python_package'] = self.godot_project.python_package

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
