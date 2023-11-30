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

    python_lib = 'python312_d'
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

pybin_copyfiles = []
if env['platform'] == 'windows':
    pybin_list = ['python312_d.dll', 'python_d.exe']
    for fn in pybin_list:
        pybin_copyfiles.append(env.Command('{0}/bin/{1}'.format(addon_path, fn),
                                           'python/PCBuild/amd64/{0}'.format(fn),
                                           Copy('$TARGET', '$SOURCE')))

env.Execute(Mkdir('{0}/lib'.format(addon_path)))
# env.Execute(Mkdir('{0}/lib/site-packages'.format(addon_path)))

pylib_copyfiles = []
pylib_list =[
    'encodings/__init__.py', 'encodings/aliases.py', 'encodings/utf_8.py', 'codecs.py',
    'io.py', 'abc.py'
]
for fn in pylib_list:
    pylib_copyfiles.append(env.Command('{0}/lib/{1}'.format(addon_path, fn),
                                        'python/Lib/{0}'.format(fn),
                                        Copy('$TARGET', '$SOURCE')))


Depends(pybin_copyfiles, library)
Depends(pylib_copyfiles, library)

Default(library, pybin_copyfiles, pylib_copyfiles)
