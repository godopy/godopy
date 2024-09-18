#!/usr/bin/env python

import os
import shutil



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
            key='minimal_python_stdlib',
            help='Copy only absolutely required part of Python standard library'
                ' to the path accessible by the game, editor will still have access'
                ' to the full library',
            default=False,
        )
    )
    opts.Add(
        BoolVariable(
            key='clear_python_files',
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


def setup_builders(env):
    import py_compile
    from binding_generator import scons_generate_bindings, scons_emit_files

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
    env.Append(CPPPATH=['src/'])
    sources = Glob('src/*.cpp')

    if env['platform'] == 'windows':
        env.Append(LIBPATH=[os.path.join('python', 'PCBuild', 'amd64')])

        python_lib = 'python312' if not env['python_debug'] else 'python312_d'
        env.Append(LIBS=[python_lib])

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
    # env_cython = env.Clone()
    # disable_warnings(env)

    generated = _generated_cython_sources(env)

    sources = [
        env.Cython('src/cythonlib/_godot.pyx'),
        env.Cython('src/cythonlib/_gdextension.pyx')
    ]

    depends = [
        generated,
        *Glob('src/cythonlib/*.pxi'),
        *Glob('src/cythonlib/*.pxd'),
        *Glob('src/cythonlib/_godot_cpp_includes/*.pxi'),
        *Glob('gdextension/*.pxd')
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
        python_dll_file = 'python312.dll' if not env['python_debug'] else 'python312_d.dll'
        python_dll = os.path.join('python', 'PCBuild', 'amd64', python_dll_file)
        python_dll_target = '{}/bin/{}/{}'.format(projectdir, env['platform'], python_dll_file)
        copy.append(env.InstallAs(python_dll_target, python_dll))

    return copy

class PythonInstaller:
    def __init__(self, env, sources, root_path):
        self.runtime_modules = set()
        self.root = root_path # os.path.join('python', 'lib')
        self.env = env
        self.sources = sources

    def install(self, files, target_lib='runtime', site_packages=False):
        projectdir = self.env['project_dir']

        for srcfile in files:
            if target_lib != 'runtime' and srcfile in self.runtime_modules:
                continue

            folder, pyfile = os.path.split(str(srcfile))
            while folder and folder.lower() != self.root:
                folder, parent = os.path.split(folder)
                pyfile = os.path.join(parent, pyfile)

            if env['compile_python_stdlib']:
                pyfile += 'c'

            if target_lib == 'runtime':
                self.runtime_modules.add(srcfile)

            real_target_lib = target_lib
            if site_packages:
                real_target_lib = os.path.join(target_lib, 'site-packages')

            dstfile = os.path.join(projectdir, 'lib', real_target_lib, pyfile)

            if env['compile_python_stdlib']:
                self.sources.append(env.CompilePyc(dstfile, srcfile))
            else:
                self.sources.append(env.InstallAs(dstfile, srcfile))

class PythonDylibInstaller:
    def __init__(self, env, sources):
        self.env = env
        self.sources = sources

    def install(self, files, target_lib='runtime'):
        projectdir = self.env['project_dir']

        for srcfile in files:
            pyfile = os.path.basename(str(srcfile))
            dstfile = os.path.join(projectdir, 'bin', 'windows', 'dylib', target_lib, pyfile)
            self.sources.append(env.InstallAs(dstfile, srcfile))


def install_python_standard_library(env):
    projectdir = env['project_dir']
    libs = []

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

    installer = PythonInstaller(env, libs, os.path.join('python', 'lib'))
    dylib_installer = PythonDylibInstaller(env, libs)

    installer.install(python_runtime_lib_files)
    installer.install(python_editor_lib_files, 'editor')
    dylib_installer.install(python_runtime_dylib_files)
    dylib_installer.install(python_editor_dylib_files, 'editor')


    libs.append(env.InstallAs(
        os.path.join(projectdir, 'lib', '.gdignore'),
        '.gdignore'
    ))

    libs.append(env.InstallAs(
        os.path.join(projectdir, 'bin', 'windows', 'dylib', '.gdignore'),
        '.gdignore'
    ))

    return libs


def install_godopy_python_libs(env):
    libs = []
    files = Glob('lib/*/*.py') + Glob('lib/*/*/*.py') + Glob('lib/*/*/*/*.py')

    installer = PythonInstaller(env, libs, 'lib')
    installer.install(files, 'runtime', True)

    return libs

###############################################################################

env = Environment(tools=['default'], PLATFORM='')
build_opts(env)

if env['clear_python_files']:
    projectdir = env['project_dir']
    shutil.rmtree(os.path.join(projectdir, 'bin', 'windows', 'dylib'), ignore_errors=True)
    shutil.rmtree(os.path.join(projectdir, 'lib'), ignore_errors=True)

# Build subdirs
Export("env")
SConscript('godot-cpp/SCsub')

setup_builders(env)

cython_env = env.Clone()

sources = main_godopy_cpp_sources(env)
sources += cython_sources(cython_env)
sources += docdata_sources(env)

library = build_extension_shared_lib(env, sources)

default_args = [
    library,
    install_extension_shared_lib(env, library),
    install_godopy_python_libs(env)
]


if env['install_python_stdlib_files']:
    default_args += install_python_standard_library(env)


Default(*default_args)
