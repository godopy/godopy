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

extension_path = 'project/addons/GodoPy/GodoPy.gdextension'
addon_path = 'project/addons/GodoPy'
project_name = 'GodoPy'

env.Append(CPPPATH=[os.path.join('python', 'Include')])

if env['platform'] == 'windows':
    env.Append(LIBPATH=[os.path.join('python', 'PCBuild', 'amd64')])
    env.Append(CPPPATH=[os.path.join('python', 'PC')])

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

# Create the library target (e.g. libGodoPy.linux.debug.x86_64.so).
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

if env['platform'] == 'windows':
    pydll = env.Command('{0}/bin/python312.dll'.format(addon_path), 'python/PCBuild/amd64/python312.dll',
                        Copy('$TARGET', '$SOURCE'))
    pyexe = env.Command('{0}/bin/python.exe'.format(addon_path), 'python/PCBuild/amd64/python.exe',
                        Copy('$TARGET', '$SOURCE'))
    venvlaunch = env.Command('{0}/bin/venvlauncher.exe'.format(addon_path), 'python/PCBuild/amd64/venvlauncher.exe',
                             Copy('$TARGET', '$SOURCE'))

    env.Execute(Mkdir('{0}/bin/py'.format(addon_path)))
    env.Execute(Mkdir('{0}/bin/edpy'.format(addon_path)))
    env.Execute(Mkdir('{0}/lib/py'.format(addon_path)))
    env.Execute(Mkdir('{0}/lib/edpy'.format(addon_path)))

    Depends(pydll, library)
    Depends(pyexe, library)
    Depends(venvlaunch, library)

Default(library, pydll, pyexe, venvlaunch)
