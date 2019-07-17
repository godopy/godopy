import os
import sys

import godot

def main():
    godot_path = detect_godot_project(*os.path.split(__file__))
    print('GODOT PROJECT PATH:', godot_path)

    if not godot_path:
        print('No Godot project found. Aborting.')
        return 1

    project_path, godot_project_name = os.path.split(godot_path)

    print('PYGODOT PROJECT PATH:', project_path)

    def fullpath(*args):
        return os.path.join(project_path, *args)

    def file_exists(*args):
        return os.path.exists(fullpath(*args))

    if not file_exists('gdlibrary.py'):
        print('No "gdlibrary.py" found. PyGodot initialization aborted.')
        return 2

    sys.path.insert(0, project_path)

    if not file_exists('__init__.py'):
        with open(fullpath('__init__.py'), 'w'):
            pass

    import gdlibrary

    if not hasattr(gdlibrary, 'nativescript_init'):
        print('GDLibrary doesn\'t provide NativeScript initialization routine. PyGodot initialization aborted.')
        return 3

    gdlibrary.nativescript_init()

def detect_godot_project(dir, fn):
    if not dir or not fn:
        return

    if 'project.godot' in os.listdir(dir):
        return dir

    return detect_godot_project(*os.path.split(dir))
