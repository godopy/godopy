#!/usr/bin/env python

import os
import shutil
import py_compile
from binding_generator import scons_generate_bindings, scons_emit_files

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

env = Environment(tools=['default'], PLATFORM='')

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
        key='minimal_python_stdlib',
        help='Copy only absolutely required part of Python standard library'
             ' to the path accessible by the game, editor will still have access'
             ' to the full library',
        default=False,
    )
)
opts.Add(
    BoolVariable(
        key='clean_python_files',
        help='Delete all previously installed files before copying Python standard library',
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

projectdir = env['project_dir']

# Build subdirs
Export("env")

SConscript('godot-cpp/SCsub')

env.Append(CPPPATH=['src/'])
sources = Glob('src/*.cpp')

if env['platform'] == 'windows':
    env.Append(LIBPATH=[os.path.join('python', 'PCBuild', 'amd64')])

    python_lib = 'python312' if not env['python_debug'] else 'python312_d'
    env.Append(LIBS=[python_lib])

    env.Append(CPPDEFINES=['WINDOWS_ENABLED'])
# TODO: Other platforms

env_cython = env.Clone()
# disable_warnings(env_cython)

cython_opts = ' '.join([
    '-3',
    '--cplus',
    '--fast-fail',
    '--gdb' if env['debug_symbols'] else '',
    '-EWITH_THREAD=1',
    '-Isrc/cythonlib',
    '-Igen/cythonlib',
    '-Igdextension',
    '-o',
    '$TARGET',
    '$SOURCE'
])

cython_builder = Builder(
    action='cython %s' % cython_opts,
    suffix='.cpp',
    src_suffox='.pyx'
)

def compile_to_bytecode(target, source, env):
    py_compile.compile(str(source[0]), str(target[0]))

env.Append(BUILDERS={
    'GodoPyBindings': Builder(action=Action(scons_generate_bindings), emitter=scons_emit_files),
    'CompilePyc': Builder(action=compile_to_bytecode)
})

env_cython.Append(BUILDERS={
    'CythonSource': cython_builder
})

extension_dir = normalize_path(env_cython.get("gdextension_dir", env.Dir("gdextension").abspath), env)
api_file = normalize_path(env_cython.get("custom_api_file", env.File(extension_dir + "/extension_api.json").abspath), env)
bindings = env.GodoPyBindings(
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

cython_sources = [
    env_cython.CythonSource('src/cythonlib/_godot.pyx'),
    env_cython.CythonSource('src/cythonlib/_gdextension.pyx')
]

cython_depends = [
    bindings,
    *Glob('src/cythonlib/*.pxi'),
    *Glob('src/cythonlib/*.pxd'),
    *Glob('src/cythonlib/_godot_cpp_includes/*.pxi'),
    *Glob('gdextension/*.pxd')
]
Depends(cython_sources, cython_depends)

sources += cython_sources

# gdextension_lib_sources = [env_cython.CythonSource('src/cythonlib/gdextension.pyx')]
# Depends(gdextension_lib_sources, cython_depends)

if env['target'] in ['editor', 'template_debug']:
    try:
        doc_data = env.GodotCPPDocData('src/gen/doc_data.gen.cpp', source=Glob('doc_classes/*.xml'))
        sources.append(doc_data)
    except AttributeError:
        print("Not including class reference as we're targeting a pre-4.3 baseline.")

file = '{}{}{}'.format(libname, env['suffix'], env['SHLIBSUFFIX'])

if env['platform'] == 'macos' or env['platform'] == 'ios':
    platlibname = '{}.{}.{}'.format(libname, env['platform'], env['target'])
    file = '{}.framework/{}'.format(env['platform'], platlibname, platlibname)

libraryfile = 'bin/{}/{}'.format(env['platform'], file)
library = env.SharedLibrary(
    libraryfile,
    source=sources,
)

# Example on how to build and install Python extensions:
# gdextension_lib_file = 'bin/{}/dylib/runtime/gdextension.pyd'.format(env['platform'])
# gdextension_lib = env.SharedLibrary(
#     gdextension_lib_file,
#     source=gdextension_lib_sources,
#     SHLIBSUFFIX='.pyd'
# )
# gdelib_filename = 'gdextension.pyd' if not env['python_debug'] else 'gdextension_d.pyd'

copy = [
    env.InstallAs('{}/bin/{}/lib{}'.format(projectdir, env['platform'], file), library),
    # env.InstallAs('{}/bin/{}/dylib/{}'.format(projectdir, env['platform'], gdelib_filename), gdextension_lib)
]

copy_python_deps = []
if env['minimal_python_stdlib']:
    # Python stdlib files - minimal version
    python_runtime_lib_files = [os.path.join('python', 'Lib', pyfile) for pyfile in [
        # These are always required
        'encodings/__init__.py', 'encodings/aliases.py', 'encodings/utf_8.py', 'codecs.py',
        'io.py', 'abc.py', 'types.py',

        'encodings/latin_1.py',
    ]]
    python_editor_lib_files = Glob('python/Lib/*.py') + Glob('python/Lib/*/*.py')
    python_editor_dylib_files = Glob('python/PCBuild/amd64/*.pyd')
else:
    python_runtime_lib_files = Glob('python/Lib/*.py') + Glob('python/Lib/*/*.py')
    python_editor_lib_files = []
    python_runtime_dylib_files = Glob('python/PCBuild/amd64/*.pyd')
    python_editor_dylib_files = []


if env['clean_python_files']:
    shutil.rmtree(os.path.join(projectdir, 'bin', 'windows', 'dylib'), ignore_errors=True)
    shutil.rmtree(os.path.join(projectdir, 'lib'), ignore_errors=True)


runtime_modules = set()
src_pylib_root = os.path.join('python', 'lib')

def install_python_files(source_files, target_lib='runtime',
                         scons_target=copy_python_deps,
                         root_path=src_pylib_root, site_packages=False):    
    for srcfile in source_files:
        if target_lib != 'runtime' and srcfile in runtime_modules:
            continue
        folder, pyfile = os.path.split(str(srcfile))
        while folder and folder.lower() != root_path:
            folder, parent = os.path.split(folder)
            pyfile = os.path.join(parent, pyfile)
        if env['compile_python_stdlib']:
            pyfile += 'c'
        if target_lib == 'runtime':
            runtime_modules.add(srcfile)
        if site_packages:
            target_lib = os.path.join(target_lib, 'site-packages')
        dstfile = os.path.join(projectdir, 'lib', target_lib, pyfile)
        if env['compile_python_stdlib']:
            scons_target.append(env.CompilePyc(dstfile, srcfile))
        else:
            scons_target.append(env.InstallAs(dstfile, srcfile))

def install_python_dylib_files(source_files, target_lib='runtime', scons_target=copy_python_deps):
    for srcfile in source_files:
        pyfile = os.path.basename(str(srcfile))
        dstfile = os.path.join(projectdir, 'bin', 'windows', 'dylib', target_lib, pyfile)
        scons_target.append(env.InstallAs(dstfile, srcfile))

install_python_files(python_runtime_lib_files)
install_python_files(python_editor_lib_files, 'editor')
install_python_dylib_files(python_runtime_dylib_files)
install_python_dylib_files(python_editor_dylib_files, 'editor')

if env['platform'] == 'windows':
    # Extension DLL requires Python DLL
    python_dll_file = 'python312.dll' if not env['python_debug'] else 'python312_d.dll'
    python_dll = os.path.join('python', 'PCBuild', 'amd64', python_dll_file)
    python_dll_target = '{}/bin/{}/{}'.format(projectdir, env['platform'], python_dll_file)
    copy_python_deps.append(env.InstallAs(python_dll_target, python_dll))

copy_python_libs = []
python_lib_files = Glob('lib/*/*.py')
install_python_files(python_lib_files, 'runtime', copy_python_libs, 'lib', True)

default_args = [library, copy, copy_python_libs]

if env['install_python_stdlib_files']:
    default_args += copy_python_deps

Default(*default_args)
