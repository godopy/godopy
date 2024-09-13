import gdextension as gde

def initialize():
    import sys
    print("Active Python paths:")
    for path in sys.path:
        print('\t', path)
    print()
