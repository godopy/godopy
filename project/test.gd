extends SceneTree

func _init():
    Python.run_simple_string(
        "import sys\n" +
        "print(sys.path)"
    )

    quit()
