from typing import Any, Mapping, Callable

import godot
from godot import classdb
from godot.types import String
from godot.classdb import (
    Engine, Node, PackedScene, Resource, ResourceLoader, ResourceSaver
)


_scene_cache = {}


class SceneMeta(type):
    def __new__(cls, name: str, bases: tuple, attrs: Mapping[str, Any], **kwargs: Any) -> type:
        root = attrs.pop('__root_class__', Node)
        path = attrs.pop('__path__', None)

        if path is None:
            path = 'res://' + String(name).to_snake_case().replace('_scene', '') + '.tscn'

        attrs['__root_class__'] = root
        attrs['__path__'] = path

        cls = super().__new__(cls, name, bases, attrs, **kwargs)

        _scene_cache[path] = cls

        return cls


class Scene(metaclass=SceneMeta):
    __root_class__ = Node

    def __init__(self, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__
            if name.endswith('Scene'):
                name = name[:-5]
            name_passed = False
        else:
            name_passed = True

        created = False
        if ResourceLoader.exists(self.__path__, 'PackedScene'):
            packed = ResourceLoader.load(self.__path__, 'PackedScene', 1)
            self.root = packed.instantiate()
            if Engine.is_editor_hint():
                # Sync scenes only when the Editor is active
                if name_passed:
                    self.root.set_name(name)

                self.name = self.root.get_name()

                for key, value in kwargs.items():
                    self.root.set(key, value)
            else:
                self.name = self.root.get_name()
        elif Engine.is_editor_hint():
            # Create scenes only when the Editor is active
            packed = PackedScene()
            self.root = self.__root_class__(name=name, **kwargs)
            self.name = name
            created = True
        else:
            self.name = name
            self.root = None
            packed = None

        res = {}

        if Engine.is_editor_hint():
            # Sync scenes only when the Editor is active

            def load_resource(key: Any, path: str, hint: str = '', cache_mode=1) -> Resource:
                resource = ResourceLoader.load(path, hint, cache_mode)
                res[key] = resource
                return resource


            if hasattr(self, 'load') and callable(self.load):
                self.load(load_resource)

            if hasattr(self, 'create') and callable(self.load):
                self.create(res.__getitem__)

            if created:
                result = packed.pack(self.root)
                if result != godot.Error.OK:
                    msg = f"An error ({godot.Error(result)!r}) occurred while packing the scene ({self.root!r})."

                    raise RuntimeError(msg)

            result = ResourceSaver.save(packed, self.__path__, 0)
            if result != godot.Error.OK:
                msg = f"An error ({godot.Error(result)!r}) occurred while saving the scene to disk."

                raise RuntimeError(msg)

            for value in res.values():
                del value

        del res
        del packed
        del self.root

        self.root = None


    @staticmethod
    def from_path(path):
        CachedClass = _scene_cache.get(path, None)
        if CachedClass is not None:
            return CachedClass()

        return None

    def get_path(self):
        return self.__path__


    def __repr__(self):
        return f"<Scene object {self.name!r} of type {self.__root_class__}>"


    def __getattr__(self, attr):
        if attr.startswith('add_') and self.root is not None:
            root = self.root
            class_name = attr[4:]  # attr[4:].title()
            cls = getattr(classdb, class_name)

            def add_child(name=None, **kwargs):
                print('Add child', cls, name, attr)
                existing = self.root.find_child(name or cls.__name__, False, True)

                if existing:
                    child = existing
                    for key, value in kwargs.items():
                        child.set(key, value)
                    print('Update existing', child)
                else:
                    child = cls(name=name, **kwargs)
                    root.add_child(child, False, 0)
                    child.set_owner(root)
                    print('Created new', child)

            return add_child

        raise AttributeError(f"{self.__class__!r} object has no attribute {attr!r}")
