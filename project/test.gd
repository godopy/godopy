extends SceneTree

func _init():
    Python.run_simple_string(
        "import sys\n" +
        "print(sys.path)\n" +
        "raise Exception('Test exc')"
    )

    Python.run_simple_string("print('Hello, world!')")

    quit()

    # Python.initialize()
    # quit(Python.run_main())
