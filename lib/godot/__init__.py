from gdextension import *

def _set_global_functions():
    # printt = UtilityFunction('printt')
    # prints = UtilityFunction('prints')

    # TODO: set all available functions dynamically

    globals()['print'] = UtilityFunction('print')
    globals()['printerr'] = UtilityFunction('printerr')
    globals()['print_verbose'] = UtilityFunction('print_verbose')
    globals()['print_rich'] = UtilityFunction('print_rich')
    globals()['printraw'] = UtilityFunction('printraw')
    globals()['push_error'] = UtilityFunction('push_error')
    globals()['push_warning'] = UtilityFunction('push_warning')


_set_global_functions()
