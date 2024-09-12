#!/usr/bin/env python
import os
import scons_methods

def normalize_path(val, env):
    return val if os.path.isabs(val) else os.path.join(env.Dir('#').abspath, val)


def validate_parent_dir(key, val, env):
    if not os.path.isdir(normalize_path(os.path.dirname(val), env)):
        raise UserError("'%s' is not a directory: %s" % (key, os.path.dirname(val)))



libname = 'GodoPy'
projectdir = 'test/project'

localEnv = Environment(tools=['default'], PLATFORM='')

customs = ['custom.py']
customs = [os.path.abspath(path) for path in customs]

opts = Variables(customs, ARGUMENTS)
opts.Add(
    BoolVariable(
        key='compiledb',
        help='Generate compilation DB (`compile_commands.json`) for external tools',
        default=localEnv.get('compiledb', False),
    )
)
opts.Add(
    PathVariable(
        key='compiledb_file',
        help='Path to a custom `compile_commands.json` file',
        default=localEnv.get('compiledb_file', 'compile_commands.json'),
        validator=validate_parent_dir,
    )
)
opts.Add(
    BoolVariable(
        key='python_debug',
        help='Use debug version of python',
        default=False,
    )
)
opts.Add(
    PathVariable(
        key='project_dir',
        help='Path to a Godot project where the extension should be installed',
        default=localEnv.get('project_dir', projectdir),
        validator=validate_parent_dir,
    )
)
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()
env['compiledb'] = False

projectdir = env['project_dir']

env.Tool('compilation_db')
compilation_db = env.CompilationDatabase(
    normalize_path(localEnv['compiledb_file'], localEnv)
)
env.Alias('compiledb', compilation_db)

env = SConscript('godot-cpp/SConstruct', {'env': env, 'customs': customs})

env.mscv = env['platform'] == 'windows'

env.Append(CPPPATH=['src/'])
sources = Glob('src/*.cpp')

if env['platform'] == 'windows':
    env.Append(LIBPATH=[os.path.join('python', 'PCBuild', 'amd64')])

    python_lib = 'python312' if not env['python_debug'] else 'python312_d'
    env.Append(LIBS=[python_lib])

    env.Append(CPPDEFINES=['WINDOWS_ENABLED'])
# TODO: Other platforms

env_cython = env.Clone()
env_cython.msvc = env.mscv
scons_methods.disable_warnings(env_cython)

cython_opts = ' '.join([
    '-3',
    '--cplus',
    '--fast-fail',
    '-Isrc/cythonlib',
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
env_cython.Append(BUILDERS={'CythonSource': cython_builder})

cython_sources = env_cython.CythonSource(['src/cythonlib/godot.pyx'])
cython_depends = [
    *Glob('src/cythonlib/*.pxi'),
    *Glob('src/cythonlib/*.pxd'),
    *Glob('src/cythonlib/_godot_cpp_includes/*.pxi'),
    *Glob('gdextension/*.pxd')
]
Depends(cython_sources, cython_depends)

sources += cython_sources

gdextension_lib_sources = env_cython.CythonSource(['src/cythonlib/gdextension.pyx'])
Depends(gdextension_lib_sources, cython_depends)

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

gdextension_lib_file = 'bin/{}/dylib/gdextension.pyd'.format(env['platform'])
gdextension_lib = env.SharedLibrary(
    gdextension_lib_file,
    source=gdextension_lib_sources,
    SHLIBSUFFIX='.pyd'
)

copy = [
    env.InstallAs('{}/bin/{}/lib{}'.format(projectdir, env['platform'], file), library),
    env.InstallAs('{}/bin/{}/dylib/gdextension.pyd'.format(projectdir, env['platform']), gdextension_lib)
]

copy_python_deps = []

# Minimal Python library
python_lib_files = [
    'encodings/__init__.py', 'encodings/aliases.py', 'encodings/utf_8.py', 'codecs.py',
    'io.py', 'abc.py', 'types.py'
]

for pyfile in python_lib_files:
    srcfile = os.path.join('python', 'Lib', pyfile)
    dstfile = os.path.join(projectdir, 'pystdlib', pyfile)
    copy_python_deps.append(env.InstallAs(dstfile, srcfile))

if env['platform'] == 'windows':
    # Extension DLL requires Python DLL
    python_dll_file = 'python312.dll' if not env['python_debug'] else 'python312_d.dll'
    python_dll = os.path.join('python', 'PCBuild', 'amd64', python_dll_file)
    python_dll_target = '{}/bin/{}/{}'.format(projectdir, env['platform'], python_dll_file)
    copy_python_deps.append(env.InstallAs(python_dll_target, python_dll))

default_args = [library, gdextension_lib, copy, copy_python_deps]
if localEnv.get('compiledb', False):
    default_args += [compilation_db]
Default(*default_args)
