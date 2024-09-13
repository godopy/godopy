import gdextension as gde

def initialize():
    import sys
    print("ActivePython paths:")
    for path in sys.path:
        print('\t', path)
    print()
