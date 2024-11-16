#!/usr/bin/env python

import os
import re
import sys
import shutil
import zipfile
from pathlib import Path


EnsureSConsVersion(4, 0)


def normalize_path(val, env):
    return val if os.path.isabs(val) else os.path.join(env.Dir('#').abspath, val)


def validate_parent_dir(key, val, env):
    if not os.path.isdir(normalize_path(os.path.dirname(val), env)):
        raise UserError("'%s' is not a directory: %s" % (key, os.path.dirname(val)))


def disable_warnings(env):
    if env['platform'] == 'windows':
        # We have to remove existing warning level defines before appending /w,
        # otherwise we get: "warning D9025 : overriding '/W3' with '/w'"
        env["CCFLAGS"] = [x for x in env["CCFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        env["CFLAGS"] = [x for x in env["CFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        env["CXXFLAGS"] = [x for x in env["CXXFLAGS"] if not (x.startswith("/W") or x.startswith("/w"))]
        env.AppendUnique(CCFLAGS=["/w"])
    else:
        env.AppendUnique(CCFLAGS=["-w"])


libname = 'GodoPy'
projectdir = 'test/project'


pythonlib_exclude_default = [
    'lib2to3',
    'idlelib',
    'test',
    'tkinter',
    'turtledemo',
    'turtle',
    'pydoc_data'
]

def build_opts(env):
    opts = Variables()

    opts.Add(
        BoolVariable(
            key='clear_pythonlib',
            help='Delete all previously installed files before copying Python standard library. ' \
                 'Always on with install_pythonlib',
            default=False,
        )
    )

    opts.Add(
        BoolVariable(
            key='install_pythonlib',
            help='Installs Python standard library',
            default=True,
        )
    )

    opts.Add(
        PathVariable(
            key='project_dir',
            help='Path to a Godot project where the extension should be installed',
            default=projectdir,
            validator=validate_parent_dir,
        )
    )

    opts.Update(env)

    Help(opts.GenerateHelpText(env))

    # TODO: Allow customization
    env['pythonlib_exclude'] = pythonlib_exclude_default


def setup_builders(env):
    import py_compile
    from binding_generator import scons_generate_bindings, scons_emit_files

    cython_opts = ' '.join([
        '-3',
        '--cplus',
        '--fast-fail',
        '--gdb' if env['debug_symbols'] else '',
        '-EWITH_THREAD=1',
        '-Isrc',
        '-Isrc/gdextension_interface',
        '-Isrc/core',
        '-Isrc/types',
        '-Isrc/godot_cpp',
        '-Igdextension',
        '-o',
        '$TARGET',
        '$SOURCE'
    ])

    def writepy(target, source, env):
        import importlib.machinery
        import importlib._bootstrap_external

        target_paths = env['python_target_paths']
        with zipfile.ZipFile(str(target[0]), 'w') as pythonlib:
            for srcfile, dstfile in zip(source, target_paths):
                loader = importlib.machinery.SourceFileLoader('<py_compile>', str(srcfile))
                source_bytes = loader.get_data(str(srcfile))
                try:
                    code = loader.source_to_code(source_bytes, str(srcfile))
                except Exception as err:
                    sys.stderr.write(f"Error during bytecode compilation of {str(dstfile)!r}: {err}\n")
                    continue

                source_stats = loader.path_stats(str(srcfile))
                bytecode = importlib._bootstrap_external._code_to_timestamp_pyc(
                    code, source_stats['mtime'], source_stats['size'])

                # print(str(srcfile), str(dstfile))
                pythonlib.writestr(str(dstfile), bytecode)

    env.Append(BUILDERS={
        'Cython': Builder(
            action='cython %s' % cython_opts,
            suffix='.cpp',
            src_suffox='.pyx'
        ),

        'GenerateBindings': Builder(
            action=Action(scons_generate_bindings),
            emitter=scons_emit_files
        ),

        'WritePy': Builder(action=writepy)
    })


def main_godopy_cpp_sources(env):
    # Entry point and Python classes
    env.AppendUnique(CPPPATH=['src/'])
    sources = Glob('src/*.cpp') + Glob('src/python/*.cpp') + Glob('src/variant/*.cpp')

    if env['platform'] == 'windows':
        env.Append(LIBPATH=[os.path.join('extern', 'cpython', 'PCBuild', 'amd64')])

        python_lib = 'python312'
        env.Append(packages=[python_lib])


    else:
        print(env['platform'])
        env.Append(LIBPATH=[os.path.join('extern', 'cpython')])
        python_lib = 'libpython3.12'
        env.Append(packages=[python_lib])

    return sources


def _generated_cython_sources(env):
    extension_dir = normalize_path(env.get("gdextension_dir", env.Dir("gdextension").abspath), env)
    api_file = normalize_path(env.get("custom_api_file", env.File(extension_dir + "/extension_api.json").abspath), env)

    bindings = env.GenerateBindings(
        env.Dir("."),
        [
            api_file,
            os.path.join(extension_dir, "gdextension_interface.h"),
            "binding_generator.py",
        ],
    )

    # Forces bindings regeneration.
    if env["generate_bindings"]:
        env.AlwaysBuild(bindings)
        env.NoCache(bindings)

    projectdir = Path(env['project_dir'])
    src_file = Path(env.Dir("#").abspath) / 'gen' / 'gdextension_interface' / 'api_data.pickle'
    target_file = projectdir / 'bin' / 'api_data.pickle'

    return [bindings, env.InstallAs(target_file, src_file)]


def cython_sources(env):
    generated = _generated_cython_sources(env)

    sources = [
        env.Cython('src/core/_gdextension_internals.pyx'),
        env.Cython('src/core/entry_point.pyx'),
        env.Cython('src/types/godot_types.pyx'),
        env.Cython('src/core/gdextension.pyx'),
    ]

    depends = [
        generated,
        *Glob('src/core/*.pxd'),
        *Glob('src/types/*.pxd'),
        *Glob('src/core/includes/*.pxi'),
        *Glob('src/types/includes/*.pxi'),
        *Glob('src/godot_cpp/*.pxd'),
        *Glob('src/godot_cpp/includes/*.pxi'),
        *Glob('gdextension/*.pxd'),
        *Glob('gen/gdextension_interface/*.pxi'),
    ]

    Depends(sources, depends)

    return sources


def docdata_sources(env):
    if env['target'] in ['editor', 'template_debug']:
        return env.GodotCPPDocData(
            'src/gen/doc_data.gen.cpp',
            source=Glob('doc_classes/*.xml')
        )
    return []


def library_file(env):
    file = '{}{}{}'.format(libname, env['suffix'], env['SHLIBSUFFIX'])

    if env['platform'] == 'macos' or env['platform'] == 'ios':
        platlibname = '{}.{}.{}'.format(libname, env['platform'], env['target'])
        file = '{}.framework/{}'.format(env['platform'], platlibname, platlibname)

    return file


def build_extension_shared_lib(env, sources):
    return env.SharedLibrary(
        'bin/{}/{}'.format(env['platform'], library_file(env)),
        source=sources,
    )


def install_extension_shared_lib(env, library):
    projectdir = env['project_dir']
    file = library_file(env)

    copy = [
        env.InstallAs(
            '{}/bin/{}/lib{}'.format(projectdir, env['platform'], file),
            library
        ),
    ]

    if env['platform'] == 'windows':
        # Extension DLL requires Python DLL
        python_dll_file = 'python312.dll'
        python_dll = os.path.join('extern', 'cpython', 'PCBuild', 'amd64', python_dll_file)
        python_dll_target = '{}/bin/{}/{}'.format(projectdir, env['platform'], python_dll_file)
        
        copy.append(env.InstallAs(python_dll_target, python_dll))

    # TODO: Other platforms

    return copy


class PythonInstaller:
    def __init__(self, env, sources, root_path):
        self.root = str(root_path)
        self.env = env
        self.sources = sources

    def get_pairs(self, files):
        pairs = []

        for srcfile in files:
            folder, pyfile = os.path.split(str(srcfile))
            while folder and folder.lower() != self.root.lower():
                folder, parent = os.path.split(folder)
                pyfile = os.path.join(parent, pyfile)

            pyfile += 'c'

            pairs.append((pyfile, srcfile))

        return pairs

class PythonDylibInstaller:
    def __init__(self, env, sources):
        self.env = env
        self.sources = sources

    def install(self, files, folder=None):
        projectdir = self.env['project_dir']

        for srcfile in files:
            pyfile = os.path.basename(str(srcfile))

            if pyfile.endswith('.pyd') and 'cp312' in pyfile:
                # Strip extra ext
                pyfile = os.path.splitext(os.path.splitext(pyfile)[0])[0] + '.pyd'

            dstfolder = os.path.join(projectdir, 'bin', 'windows')

            if folder is not None:
                dstfolder = os.path.join(dstfolder, folder)

            dstfile = os.path.join(dstfolder, pyfile)
            self.sources.append(env.InstallAs(dstfile, srcfile))

def install_python_standard_library(env):
    packages = []

    python_lib_files = Glob('extern/cpython/Lib/*.py') + Glob('extern/cpython/Lib/*/*.py')
    if env['platform'] == 'windows':
        python_dylib_files = [
            f for f in Glob('extern/cpython/PCBuild/amd64/*.pyd')
            if 'test' not in str(f) and 'tkinter' not in str(f) and 'xxlimited' not in str(f)
        ]
    else:
        print(env['platform'])
        python_dylib_files = [
            f for f in Glob('extern/cpython/build/lib.linux-x86_64-3.12/*.so')
            if 'test' not in str(f) and 'tkinter' not in str(f) and 'xxlimited' not in str(f)
        ]

    files_filtered = []

    for f in python_lib_files:
        for path in env['pythonlib_exclude']:
            if f.relpath[19:].startswith(path):
                break
        else:
            files_filtered.append(f)

    installer = PythonInstaller(env, packages, os.path.join('extern', 'cpython', 'lib'))
    dylib_installer = PythonDylibInstaller(env, packages)

    dylib_installer.install(python_dylib_files)

    return packages, installer.get_pairs(files_filtered)


def install_extra_python_packages(env):
    if env['platform'] == 'windows':
        numpy_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy'
        numpylibs_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy.libs'
    else:
        numpy_folder = Path(venv_folder) / 'lib' / 'python3.12' / 'site-packages' / 'numpy'
        numpylibs_folder = Path(venv_folder) / 'lib' / 'python3.12' / 'site-packages' / 'numpy.libs'

    build_path = env.Dir('#').abspath

    packages = []

    files = [
        str(f).replace(str(build_path), '') for f in [
            *Glob(str(numpy_folder / '*.py')),
            *Glob(str(numpy_folder / '*' / '*.py')),
            *Glob(str(numpy_folder / '*' / '*' / '*.py')),
            *Glob(str(numpy_folder / '*' / '*' / '*' / '*.py')),
            *Glob(str(numpy_folder / '*' / '*' / '*' / '*' / '*.py'))
        ] if f'{os.sep}tests{os.sep}' not in str(f)
             and f'{os.sep}testing{os.sep}' not in str(f)
             and f'{os.sep}_examples{os.sep}' not in str(f)
             and f'{os.sep}f2py{os.sep}' not in str(f)
    ]

    if env['platform'] == 'windows':
        dylib_files = [str(f).replace(str(build_path), '') for f in [
            *Glob(str(numpy_folder / '*' / '*.pyd')),
            *Glob(str(numpylibs_folder / '*.dll'))
        ] if '_tests' not in str(f)]
    else:
        dylib_files = [str(f).replace(str(build_path), '') for f in [
            *Glob(str(numpy_folder / '*' / '*.so')),
            *Glob(str(numpylibs_folder / '*.so'))
        ] if '_tests' not in str(f)]

    root = Path(str(files[0]).split('site-packages')[0]) / 'site-packages'

    installer = PythonInstaller(env, packages, root)
    dylib_installer = PythonDylibInstaller(env, packages)

    dylib_installer.install(dylib_files)

    return packages, installer.get_pairs(files)


def install_godopy_python_packages(env):
    files = Glob('lib/*/*.py') + Glob('lib/*/*/*.py') + Glob('lib/*/*/*/*.py')

    installer = PythonInstaller(env, packages, 'lib')

    return installer.get_pairs(files)

###############################################################################

if not 'VIRTUAL_ENV' in os.environ:
    raise Exception("No virtual environment detected. "
                    "Please create and/or activate one "
                    "and install all requirements from 'requirements.txt'")

venv_path = os.environ['VIRTUAL_ENV']
if venv_path.startswith('/cygdrive/c'):
    # Fix Cygwin paths
    venv_path = re.sub(r"^/cygdrive/(\w)/", lambda m: f"{m.group(1).upper()}:/", venv_path)

venv_folder = os.path.abspath(venv_path)
numpy_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy'
numpylibs_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy.libs'


env = Environment(tools=['default'], PLATFORM='')
build_opts(env)

if env['install_pythonlib']:
    env['clear_pythonlib'] = True


if not numpy_folder.is_relative_to(env.Dir('#').abspath):
    raise Exception("Virtual Env folder must be located inside the current folder")


if env['clear_pythonlib']:
    projectdir = env['project_dir']
    shutil.rmtree(os.path.join(projectdir, 'bin', 'windows'), ignore_errors=True)
    shutil.rmtree(os.path.join(projectdir, 'bin', 'pythonlib.zip'), ignore_errors=True)
    shutil.rmtree(os.path.join(projectdir, 'bin', 'api_data.pickle'), ignore_errors=True)


# Build subdirs
Export("env")
SConscript('godot-cpp/SCsub')

setup_builders(env)

if env.dev_build:
    # Use local GodoPy Python libs in dev mode
    env.Append(
        CPPDEFINES=(
            'GODOPY_LIB_PATH',
            os.path.join(env.Dir('#').abspath, 'lib').replace(os.sep, '/')
        )
    )

cython_env = env.Clone()

sources = main_godopy_cpp_sources(env)
sources += cython_sources(cython_env)
sources += docdata_sources(env)

library = build_extension_shared_lib(env, sources)
env.NoCache(library)

default_args = [
    library,
    install_extension_shared_lib(env, library)
]


pythonlib_files = []

if not env.dev_build:
    files = install_godopy_python_packages(env)
    pythonlib_files += files

if env['install_pythonlib']:
    dylib_install, files = install_python_standard_library(env)
    default_args += dylib_install
    pythonlib_files += files

    dylib_install, files = install_extra_python_packages(env)
    default_args += dylib_install
    pythonlib_files += files

if pythonlib_files:
    pythonlib = Path(env['project_dir']) / 'bin' / 'pythonlib.zip'

    files = []
    targets = []
    for dst, src in pythonlib_files:
        files.append(src)
        targets.append(dst)
    env['python_target_paths'] = targets

    default_args += [env.WritePy(pythonlib, files)]


Default(*default_args)
