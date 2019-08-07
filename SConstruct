#!python

import sys
import os

if not hasattr(sys, 'version_info') or sys.version_info < (3, 7):
    raise SystemExit("PyGodot requires Python version 3.7 or above.")

# Try to detect the host platform automatically.
# This is used if no `platform` argument is passed
if sys.platform.startswith('linux'):
    host_platform = 'linux'
elif sys.platform == 'darwin':
    host_platform = 'osx'
elif sys.platform == 'win32' or sys.platform == 'msys':
    host_platform = 'windows'
else:
    raise ValueError(
        'Could not detect platform automatically, please specify with '
        'platform=<platform>'
    )

opts = Variables([], ARGUMENTS)
opts.Add(EnumVariable(
    'platform',
    'Target platform',
    host_platform,
    allowed_values=('linux', 'osx', 'windows'),
    ignorecase=2
))
opts.Add(EnumVariable(
    'bits',
    'Target platform bits',
    'default',
    ('default', '32', '64')
))
opts.Add(BoolVariable(
    'use_llvm',
    'Use the LLVM compiler - only effective when targeting Linux',
    False
))
opts.Add(BoolVariable(
    'use_mingw',
    'Use the MinGW compiler instead of MSVC - only effective on Windows',
    False
))
# Must be the same setting as used for cpp_bindings
opts.Add(EnumVariable(
    'target',
    'Compilation target',
    'debug',
    allowed_values=('debug', 'release'),
    ignorecase=2
))
opts.Add(
    'target_extension',
    'Python GDNative extension path',
    ''
)
opts.Add(PathVariable(
    'headers_dir',
    'Path to the directory containing Godot headers',
    'godot_headers',
    PathVariable.PathIsDir
))
opts.Add(PathVariable(
    'custom_api_file',
    'Path to a custom JSON API file',
    None,
    PathVariable.PathIsFile
))
opts.Add(BoolVariable(
    'generate_bindings',
    'Generate GDNative API bindings',
    False
))

env = Environment()
opts.Update(env)
Help(opts.GenerateHelpText(env))

is64 = sys.maxsize > 2**32
if (
    env['TARGET_ARCH'] == 'amd64' or
    env['TARGET_ARCH'] == 'emt64' or
    env['TARGET_ARCH'] == 'x86_64'
):
    is64 = True

if env['bits'] == 'default':
    env['bits'] = '64' if is64 else '32'

# This makes sure to keep the session environment variables on Windows.
# This way, you can run SCons in a Visual Studio 2017 prompt and it will find
# all the required tools
if host_platform == 'windows':
    if env['bits'] == '64':
        env = Environment(TARGET_ARCH='amd64')
    elif env['bits'] == '32':
        env = Environment(TARGET_ARCH='x86')

    opts.Update(env)

if host_platform == 'osx':
    env.Append(LIBPATH=[os.path.join('buildenv', 'lib', 'python3.8', 'config-3.8-darwin')])
env.Append(CPPPATH=[os.path.join('buildenv', 'include', 'python3.8')])

if env['platform'] == 'linux':
    if env['use_llvm']:
        env['CXX'] = 'clang++'

    env.Append(CCFLAGS=['-fPIC', '-g', '-std=c++14', '-Wwrite-strings'])
    env.Append(LINKFLAGS=["-Wl,-R,'$$ORIGIN'"])

    if env['target'] == 'debug':
        env.Append(CCFLAGS=['-Og'])
    elif env['target'] == 'release':
        env.Append(CCFLAGS=['-O3'])

    if env['bits'] == '64':
        env.Append(CCFLAGS=['-m64'])
        env.Append(LINKFLAGS=['-m64'])
    elif env['bits'] == '32':
        env.Append(CCFLAGS=['-m32'])
        env.Append(LINKFLAGS=['-m32'])

elif env['platform'] == 'osx':
    # Use Clang on macOS by default
    env['CXX'] = 'clang++'

    if env['bits'] == '32':
        raise ValueError(
            'Only 64-bit builds are supported for the macOS target.'
        )

    env.Append(CCFLAGS=['-g', '-std=c++14', '-arch', 'x86_64', '-fwrapv', '-Wno-unused-result', '-Wsign-compare'])
    env.Append(LINKFLAGS=[
        '-arch',
        'x86_64',
        '-framework', 'Cocoa',
        '-Wl,-undefined,dynamic_lookup',
    ])

    env.Append(LIBS=['dl'])

    if env['target'] == 'debug':
        env.Append(CCFLAGS=['-Og'])
    elif env['target'] == 'release':
        env.Append(CCFLAGS=['-O3'])

elif env['platform'] == 'windows':
    if host_platform == 'windows' and not env['use_mingw']:
        # MSVC
        env.Append(LINKFLAGS=['/WX'])
        if env['target'] == 'debug':
            env.Append(CCFLAGS=['/Z7', '/Od', '/EHsc', '/D_DEBUG', '/MDd', '/bigobj'])
        elif env['target'] == 'release':
            env.Append(CCFLAGS=['/O2', '/EHsc', '/DNDEBUG', '/MD', '/bigobj'])

    elif host_platform == 'linux' or host_platform == 'osx':
        # Cross-compilation using MinGW
        if env['bits'] == '64':
            env['CXX'] = 'x86_64-w64-mingw32-g++'
            env['AR'] = "x86_64-w64-mingw32-ar"
            env['RANLIB'] = "x86_64-w64-mingw32-ranlib"
            env['LINK'] = "x86_64-w64-mingw32-g++"
        elif env['bits'] == '32':
            env['CXX'] = 'i686-w64-mingw32-g++'
            env['AR'] = "i686-w64-mingw32-ar"
            env['RANLIB'] = "i686-w64-mingw32-ranlib"
            env['LINK'] = "i686-w64-mingw32-g++"

    # Native or cross-compilation using MinGW
    if host_platform == 'linux' or host_platform == 'osx' or env['use_mingw']:
        env.Append(CCFLAGS=['-g', '-O3', '-std=c++14', '-Wwrite-strings'])
        env.Append(LINKFLAGS=[
            '--static',
            '-Wl,--no-undefined',
            '-static-libgcc',
            '-static-libstdc++',
        ])

binpath = os.path.dirname(sys.executable)
if sys.platform == 'win32':
    binpath = os.path.join(sys.prefix, 'Scripts')

env.Append(BUILDERS={
    'CythonSource': Builder(action='%s/cython --fast-fail -3 --cplus -o $TARGET $SOURCE' % binpath)
})

env.Append(CPPPATH=[
    '.',
    env['headers_dir'],
    'include',
    'include/gen',
    'include/core',
    'include/pycore',
    'include/pygen',
])

# Generate bindings?
json_api_file = ''

if 'custom_api_file' in env:
    json_api_file = env['custom_api_file']
else:
    json_api_file = os.path.join(os.getcwd(), 'godot_headers', 'api.json')

if env['generate_bindings']:
    # Actually create the bindings here
    import binding_generator

    binding_generator.generate_bindings(json_api_file)

# Sources to compile
cython_sources = [env.CythonSource(str(fp).replace('.pyx', '.cpp'), fp) for fp in Glob('godot/*.pyx')]
cython_extra_sources = [env.CythonSource(str(fp).replace('.pyx', '.cpp'), fp) for fp in Glob('pygodot/*.pyx')]
cython_binding_sources = [env.CythonSource(str(fp).replace('.pyx', '.cpp'), fp) for fp in Glob('godot/bindings/*.pyx')]

sources = [
    *cython_sources,
    *cython_extra_sources,
    *cython_binding_sources,
    *Glob('src/core/*.cpp'),
    *Glob('src/gen/*.cpp'),
    *Glob('src/pycore/*.cpp')
]

gdlib_sources = [
    'src/pylib/gdlibrary.cpp'
]

static_target_name = 'bin/libpygodot.%(platform)s.%(target)s.%(bits)s' % env

if env['target_extension']:
    # Library name was generated by setuptools
    static_target_name, _ext = os.path.splitext(env['target_extension'])
    static_target_name = static_target_name + env['LIBSUFFIX']

static_library = env.StaticLibrary(target=static_target_name, source=sources)

if env['target_extension']:
    Default(static_library)
    # dl_env = env.Clone()
    # dl_env.Append(LIBS=[static_library, 'python37']) # XXX win32 name, python3.7m on mac
    # dl_env['SHLIBPREFIX'] = ''
    # dl_env['SHLINKFLAGS'] = ''
    # dl_env['SHLIBSUFFIX'] = '.pyd'
    # gdlib = dl_env.SharedLibrary(target=env['target_extension'], source=gdlib_sources)

    # Default(gdlib)
else:
    Default(static_library)
