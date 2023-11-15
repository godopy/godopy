#!/usr/bin/env python
import os
from glob import glob
from pathlib import Path

env = SConscript("godot-cpp/SConstruct")

env.Append(CPPPATH=["extension/src/"])
sources = Glob("extension/src/*.cpp")

env.Append(BUILDERS={
    'CythonSource': Builder(
        action='cython --fast-fail -3 --cplus -o $TARGET $SOURCE'
    )
})

# Find gdextension path even if the directory or extension is renamed (e.g. project/addons/example/example.gdextension).
(extension_path,) = glob("project/addons/*/*.gdextension")

# Find the addon path (e.g. project/addons/example).
addon_path = Path(extension_path).parent

# Find the project name from the gdextension file (e.g. example).
project_name = Path(extension_path).stem

if env['platform'] == 'windows':
    env.Append(LIBPATH=[os.path.join('python', 'PCBuild', 'amd64')])
    env.Append(CPPPATH=[os.path.join('python', 'PC')])
    env.Append(CPPPATH=[os.path.join('python', 'Include')])

    env.Append(LINKFLAGS=['/WX'])
    # if env['target'] == 'debug':
    #     env.Append(CCFLAGS=['/Z7', '/Od', '/EHsc', '/D_DEBUG', '/MDd', '/bigobj'])
    # elif env['target'] == 'release':
    env.Append(CCFLAGS=['/O2', '/EHsc', '/DNDEBUG', '/MD', '/bigobj'])

    python_lib = 'python312'
    env.Append(LIBS=[python_lib])

    env.Append(CPPDEFINES=['WINDOWS_ENABLED'])

lib_dir = 'extension/src/lib/'
lib_sources = [
    '_godopy_bootstrap'
]

lib_sources = [
    env.CythonSource(
        '%s%s.cpp' % (lib_dir, f),
        '%s%s.pyx' % (lib_dir, f)
    ) for f in lib_sources
]

# Create the library target (e.g. libexample.linux.debug.x86_64.so).
debug_or_release = "release" if env["target"] == "template_release" else "debug"
if env["platform"] == "macos":
    library = env.SharedLibrary(
        "{0}/bin/lib{1}.{2}.{3}.framework/{1}.{2}.{3}".format(
            addon_path,
            project_name,
            env["platform"],
            debug_or_release,
        ),
        sources + lib_sources,
    )
else:
    library = env.SharedLibrary(
        "{}/bin/lib{}.{}.{}.{}{}".format(
            addon_path,
            project_name,
            env["platform"],
            debug_or_release,
            env["arch"],
            env["SHLIBSUFFIX"],
        ),
        sources + lib_sources,
    )

# TODO: Copy python312.dll on windows to the

Default(library)
