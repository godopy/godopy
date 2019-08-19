#!python

import sys
import os

if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    raise SystemExit("PyGodot requires Python version 3.6 or above.")

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
opts.Add(BoolVariable(
    'only_cython',
    'Compile only Cython sources',
    False
))
opts.Add(BoolVariable(
    'python_debug',
    'Use debug build of Python',
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
    'internal-packages/godot_headers',
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
if env['TARGET_ARCH'] == 'amd64' or env['TARGET_ARCH'] == 'emt64' or env['TARGET_ARCH'] == 'x86_64':
    is64 = True

if env['bits'] == 'default':
    env['bits'] = '64' if is64 else '32'

python_include = 'python3.8d' if env['python_debug'] else 'python3.8'
python_lib = 'python3.8d' if env['python_debug'] else 'python3.8'
python_internal_env = os.path.join('buildenv', 'lib', 'python3.8', 'site-packages')

# This makes sure to keep the session environment variables on Windows.
# This way, you can run SCons in a Visual Studio 2017 prompt and it will find
# all the required tools
if host_platform == 'windows':
    bits = env['bits']

    if env['bits'] == '64':
        env = Environment(TARGET_ARCH='amd64')
    elif env['bits'] == '32':
        env = Environment(TARGET_ARCH='x86')

    opts.Update(env)
    env['bits'] = bits
    python_internal_env = os.path.join('buildenv', 'Lib', 'site-packages')

if env['platform'] == 'linux':
    if env['use_llvm']:
        env['CXX'] = 'clang++'

    # libdir = 'config-3.8d-darwin' if env['target'] == 'debug' else 'config-3.8-darwin'
    env.Append(LIBPATH=[os.path.join('deps', 'python', 'build', 'lib')])
    env.Append(CPPPATH=[os.path.join('deps', 'python', 'build', 'include', python_include)])

    env.Append(CCFLAGS=[
        '-fPIC',
        '-g',
        '-std=c++14',
        '-Wwrite-strings',
        '-fwrapv',
        '-Wno-unused-result',
        '-Wsign-compare'
    ])
    env.Append(LINKFLAGS=["-Wl,-R,'$$ORIGIN'"])

    env.Append(LIBS=[python_lib, 'crypt', 'pthread', 'dl', 'util', 'm'])

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

    libdir = 'config-3.8d-darwin' if env['python_debug'] else 'config-3.8-darwin'
    env.Append(LIBPATH=[os.path.join('deps', 'python', 'build', 'lib', 'python3.8', libdir)])
    env.Append(CPPPATH=[os.path.join('deps', 'python', 'build', 'include', python_include)])

    if env['bits'] == '32':
        raise ValueError(
            'Only 64-bit builds are supported for the macOS target.'
        )

    env.Append(CCFLAGS=[
        '-g',
        '-std=c++14',
        '-arch', 'x86_64',
        '-fwrapv',
        '-Wno-unused-result',
        '-Wsign-compare',
        # '-Wunreachable-code'
    ])
    env.Append(LINKFLAGS=[
        '-arch',
        'x86_64',
        '-framework', 'Cocoa',
        '-Wl,-undefined,dynamic_lookup',
    ])

    env.Append(LIBS=[python_lib, 'dl'])

    if env['target'] == 'debug':
        env.Append(CCFLAGS=['-Og'])
    elif env['target'] == 'release':
        env.Append(CCFLAGS=['-O3'])

elif env['platform'] == 'windows':
    env.Append(LIBPATH=[os.path.join('deps', 'python', 'PCBuild', 'amd64')])
    env.Append(CPPPATH=[os.path.join('deps', 'python', 'PC')])
    env.Append(CPPPATH=[os.path.join('deps', 'python', 'Include')])

    python_lib = 'python38_d' if env['python_debug'] else 'python38'
    env.Append(LIBS=[python_lib])

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

binpath = os.path.join('buildenv', 'Scripts' if sys.platform == 'win32' else 'bin')

env.Append(BUILDERS={
    # 'CythonSource': Builder(action='%s/cython --fast-fail -3 --cplus -o $TARGET $SOURCE' % binpath),
    'CythonSource': Builder(action='%s/pygodot_cython $SOURCE $TARGET' % binpath)
})

env.Append(CPPPATH=[
    '.',
    env['headers_dir'],
    'include',
    'include/gen',
    'include/core',
    'include/pycore',
    'include/pygen',
    os.path.join(python_internal_env, 'numpy', 'core', 'include')
])

# Generate bindings?
json_api_file = ''

if 'custom_api_file' in env:
    json_api_file = env['custom_api_file']
else:
    json_api_file = os.path.join(os.getcwd(), 'internal-packages', 'godot_headers', 'api.json')

if env['generate_bindings']:
    # Actually create the bindings here
    import binding_generator

    binding_generator.generate_bindings(json_api_file)

# Sources to compile
cython_sources = [env.CythonSource(str(fp).replace('.pyx', '.cpp'), fp) for fp in Glob('internal-packages/godot/*.pyx')]
cython_core_sources = [env.CythonSource(str(fp).replace('.pyx', '.cpp'), fp) for fp in Glob('internal-packages/godot/core/*.pyx')]
cython_binding_sources = [env.CythonSource(str(fp).replace('.pyx', '.cpp'), fp) for fp in Glob('internal-packages/godot/bindings/*.pyx')]

sources = cython_sources + cython_core_sources + cython_binding_sources

if not env['only_cython']:
    sources += [*Glob('src/core/*.cpp'), *Glob('src/gen/*.cpp'), *Glob('src/pycore/*.cpp')]

# gdlib_sources = [
#     *cython_sources,
#     *cython_binding_sources,
#     'src/pylib/gdlibrary.cpp'
# ]

static_target_name = 'bin/libpygodot.%(platform)s.%(target)s.%(bits)s%(LIBSUFFIX)s' % env

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
