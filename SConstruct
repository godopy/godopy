#!/usr/bin/env python

import os
import shutil
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


def build_opts(env):
    opts = Variables()
    opts.Add(
        BoolVariable(
            key='python_debug',
            help='Use debug version of python',
            default=False,
        )
    )
    opts.Add(
        BoolVariable(
            key='clear_python_files',
            help='Delete all previously installed files before copying Python standard library. ' \
                 'Always on with install_python_stdlib_files',
            default=False,
        )
    )
    opts.Add(
        BoolVariable(
            key='install_python_stdlib_files',
            help='Installs Python standard library',
            default=True,
        )
    )
    opts.Add(
        BoolVariable(
            key='compile_python_stdlib',
            help='Compile library Python files to byte-code .pyc files',
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
        '-Igen/gdextension_interface',
        '-Isrc/core',
        '-Isrc/godot_cpp',
        '-Igdextension',
        '-o',
        '$TARGET',
        '$SOURCE'
    ])

    def compile_to_bytecode(target, source, env):
        py_compile.compile(str(source[0]), str(target[0]))

    env.Append(BUILDERS={
        'Cython': Builder(
            action='cython %s' % cython_opts,
            suffix='.cpp',
            src_suffox='.pyx'
        ),

        'GenerateBindings': Builder
            (action=Action(scons_generate_bindings),
             emitter=scons_emit_files
        ),

        'CompilePyc': Builder(action=compile_to_bytecode)
    })


def main_godopy_cpp_sources(env):
    # Entry point and Python classes
    env.AppendUnique(CPPPATH=['src/'])
    sources = Glob('src/*.cpp') + Glob('src/python/*.cpp') + Glob('src/variant/*.cpp')

    if env['platform'] == 'windows':
        env.Append(LIBPATH=[os.path.join('python', 'PCBuild', 'amd64')])

        python_lib = 'python312_d' if env['python_debug'] else 'python312'
        env.Append(packages=[python_lib])
        env.Append(CPPDEFINES=['WINDOWS_ENABLED'])


    # TODO: Other platforms

    return sources


def _generated_cython_sources(env):
    extension_dir = normalize_path(env.get("gdextension_dir", env.Dir("gdextension").abspath), env)
    api_file = normalize_path(env.get("custom_api_file", env.File(extension_dir + "/extension_api.json").abspath), env)

    bindings =  env.GenerateBindings(
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

    return bindings


def cython_sources(env):
    generated = _generated_cython_sources(env)

    # always required:
    # 'encodings/__init__.py', 'encodings/aliases.py', 'encodings/utf_8.py', 'codecs.py',
    # 'io.py', 'abc.py', 'types.py',
    # 'encodings/latin_1.py',

    sources = [
        env.Cython('src/core/gdextension.pyx'),
        env.Cython('src/core/entry_point.pyx'),
        env.Cython('src/core/godot_types.pyx'),
    ]

    depends = [
        generated,
        *Glob('src/core/*.pxd'),
        *Glob('src/core/*_includes/*.pxi'),
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
        python_dll_file = 'python312_d.dll' if env['python_debug'] else 'python312.dll'
        python_dll = os.path.join('python', 'PCBuild', 'amd64', python_dll_file)
        python_dll_target = '{}/bin/{}/{}'.format(projectdir, env['platform'], python_dll_file)
        
        copy.append(env.InstallAs(python_dll_target, python_dll))

    # TODO: Other platforms

    return copy

class PythonInstaller:
    def __init__(self, env, sources, root_path):
        self.root = str(root_path)
        self.env = env
        self.sources = sources

    def install(self, files, site_packages=False, force_install=False):
        projectdir = self.env['project_dir']

        for srcfile in files:
            folder, pyfile = os.path.split(str(srcfile))
            while folder and folder.lower() != self.root.lower():
                folder, parent = os.path.split(folder)
                pyfile = os.path.join(parent, pyfile)

            if env['compile_python_stdlib'] and not force_install:
                pyfile += 'c'

            target_lib = 'lib'
            if site_packages:
                target_lib = os.path.join(target_lib, 'site-packages')

            dstfile = os.path.join(projectdir, 'python', 'windows', target_lib, pyfile)

            if env['compile_python_stdlib'] and not force_install:
                self.sources.append(env.CompilePyc(dstfile, srcfile))
            else:
                self.sources.append(env.InstallAs(dstfile, srcfile))

class PythonDylibInstaller:
    def __init__(self, env, sources):
        self.env = env
        self.sources = sources

    def install(self, files, folder=None):
        projectdir = self.env['project_dir']

        for srcfile in files:
            pyfile = os.path.basename(str(srcfile))
            dstfolder = os.path.join(projectdir, 'python', 'windows', 'lib')
            if folder is not None:
                dstfolder = os.path.join(dstfolder, folder)
            dstfile = os.path.join(dstfolder, pyfile)
            self.sources.append(env.InstallAs(dstfile, srcfile))


def install_python_standard_library(env):
    projectdir = env['project_dir']
    packages = []

    python_lib_files = Glob('python/Lib/*.py') + Glob('python/Lib/*/*.py')
    python_dylib_files = Glob('python/PCBuild/amd64/*.pyd')

    installer = PythonInstaller(env, packages, os.path.join('python', 'lib'))
    dylib_installer = PythonDylibInstaller(env, packages)

    installer.install(python_lib_files)
    dylib_installer.install(python_dylib_files)

    packages.append(env.InstallAs(
        os.path.join(projectdir, 'python', '.gdignore'),
        '.gdignore'
    ))

    return packages


def install_extra_python_packages(env):
    numpy_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy'
    numpylibs_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy.libs'

    build_path = env.Dir('#').abspath

    packages = []

    files = [
        str(f).replace(str(build_path), '') for f in [
            *Glob(str(numpy_folder / '*.py')),
            *Glob(str(numpy_folder / '*' / '*.py')),
            *Glob(str(numpy_folder / '*' / '*' / '*.py')),
            *Glob(str(numpy_folder / '*' / '*' / '*' / '*.py')),
            *Glob(str(numpy_folder / '*' / '*' / '*' / '*' / '*.py'))
        ]
    ]

    dylib_files = [str(f).replace(str(build_path), '') for f in  [
        *Glob(str(numpy_folder / '*' / '*.pyd')),
        *Glob(str(numpylibs_folder / '*.dll'))
    ]]

    root = Path(str(files[0]).split('site-packages')[0]) / 'site-packages'

    installer = PythonInstaller(env, packages, root)
    # dylib_installer = PythonDylibInstaller(env, packages)

    installer.install(files, True)
    installer.install(dylib_files, True, force_install=True)

    return packages


def install_godopy_python_packages(env):
    packages = []
    files = Glob('lib/*/*.py') + Glob('lib/*/*/*.py') + Glob('lib/*/*/*/*.py')

    installer = PythonInstaller(env, packages, 'lib')
    installer.install(files, True)

    return packages

###############################################################################

if not 'VIRTUAL_ENV' in os.environ and not os.path.exists('./venv'):
    raise Exception("No virtual environment detected. "
                    "Please create and/or activate one "
                    "and install all requirements from 'requirements.txt'")

venv_folder = os.path.abspath(os.environ.get('VIRTUAL_ENV', 'venv'))
numpy_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy'
numpylibs_folder = Path(venv_folder) / 'Lib' / 'site-packages' / 'numpy.libs'


env = Environment(tools=['default'], PLATFORM='')
build_opts(env)

if env['install_python_stdlib_files']:
    env['clear_python_files'] = True

if not numpy_folder.is_relative_to(env.Dir('#').abspath):
    raise Exception("Virtual Env folder must be located inside the current folder")


if env['clear_python_files']:
    projectdir = env['project_dir']
    shutil.rmtree(os.path.join(projectdir, 'python'), ignore_errors=True)


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

if not env.dev_build:
    default_args += install_godopy_python_packages(env)

if env['install_python_stdlib_files']:
    default_args += install_python_standard_library(env)
    default_args += install_extra_python_packages(env)


Default(*default_args)
