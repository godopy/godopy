import types
import builtins
import gdextension as gde


input = gde.input


def _set_global_functions():
    # printt = UtilityFunction('printt')
    # prints = UtilityFunction('prints')

    # TODO: set all available functions dynamically

    globals()['print'] = gde.UtilityFunction('print')
    globals()['printerr'] = gde.UtilityFunction('printerr')
    globals()['print_verbose'] = gde.UtilityFunction('print_verbose')
    globals()['print_rich'] = gde.UtilityFunction('print_rich')
    globals()['printraw'] = gde.UtilityFunction('printraw')
    globals()['push_error'] = gde.UtilityFunction('push_error')
    globals()['push_warning'] = gde.UtilityFunction('push_warning')


MODULE_INITIALIZATION_LEVEL_CORE = 0
MODULE_INITIALIZATION_LEVEL_SERVERS = 1
MODULE_INITIALIZATION_LEVEL_SCENE = 2
MODULE_INITIALIZATION_LEVEL_EDITOR = 3


_set_global_functions()


class GodotClassBase(type):
    """Metaclass for all Godot engine and extension classes"""

    def __new__(cls, name, bases, attrs, **kwargs):
        super_new = super().__new__

        godot_cls = attrs.get('__godot_class__', None)
        if godot_cls is not None:
            # Engine class
            return super_new(cls, name, bases, attrs)

        # Ensure initialization is only performed for subclasses of Godot classes
        # (excluding Class/EngineClass).
        parents = [b for b in bases if isinstance(b, GodotClassBase)]
        if not parents:
            if name != 'Class' and name != 'EngineClass':
                push_warning("Attempt to create %r Godot class without a valid base" % name)
            return super_new(cls, name, bases, attrs)

        godot_cls = gde.ExtensionClass(name, parents[0].__godot_class__)

        module = attrs.pop('__module__')
        new_attrs = {
            '__module__': module,
            '__godot_class__': godot_cls
        }

        cls.register = godot_cls.register
        cls.register_abstract = godot_cls.register_abstract
        cls.register_internal = godot_cls.register_internal
        cls.register_runtime = godot_cls.register_runtime

        for attr, value in attrs.items():
            parent_method_info = godot_cls.__inherits__.get_method_info(attr)
            if attr == '__init__':
                new_attrs[attr] = godot_cls.bind_python_method(value)
            elif attr.startswith('_') and parent_method_info is not None:
                builtins.print('Meta: FOUND VIRTUAL', attr, parent_method_info)
                new_attrs[attr] = godot_cls.bind_virtual_method(value)
            elif getattr(value, '_gdmethod', False):
                new_attrs[attr] = godot_cls.bind_method(value)
            elif getattr(value, '_gdvirtualmethod', False):
                new_attrs[attr] = godot_cls.add_virtual_method(value)
            elif isinstance(value, types.FunctionType):
                new_attrs[attr] = godot_cls.bind_python_method(value)
            else:
                new_attrs[attr] = value

        return super_new(cls, name, bases, new_attrs, **kwargs)


class Extension(gde.Extension):
    def __getattr__(self, name):
        mb = gde.MethodBind(self, name)

        self.__dict__[name] = mb

        return mb

    def __dir__(self):
        return list(set(
            self.__godot_class__.__method_info__.keys() +
            self.__dict__.keys()
        ))


class ExtensionClass(gde.ExtensionClass):
    def __call__(self):
        if self.is_registered:
            return Extension(self, self.__inherits__)

class Class(Extension, metaclass=GodotClassBase):
    def __init__(self):
        godot_cls = self.__class__.__godot_class__
        super().__init__(godot_cls, godot_cls.__inherits__)


class EngineObject(gde.Object):
    def __getattr__(self, name):
        mb = gde.MethodBind(self, name)

        self.__dict__[name] = mb

        return mb


    def __dir__(self):
        return self.__godot_class__.__method_info__.keys()


class EngineClass(EngineObject, metaclass=GodotClassBase):
    def __init__(self):
        godot_cls = self.__class__.__godot_class__
        super().__init__(godot_cls)
