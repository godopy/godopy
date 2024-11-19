from typing import Any, Mapping, Callable

import godot
from godot import classdb
from godot.types import String
from godot.classdb import (
    Engine, Node as _Node, PackedScene, Resource, ResourceLoader, ResourceSaver
)


_scene_cache = {}



class Node:
    def __init__(self, node_class: godot.GodotClassBase, **properties: Any) -> None:
        self.node_class = node_class
        self._name = None
        self.properties = properties

    def set_name(self, name: str) -> str:
        self._name = name
        return name

    def get_name(self) -> str:
        if self._name is None:
            raise ValueError("scene.Node name is not set")
        return self._name

    name = property(get_name, set_name)


class ExtResource:
    def __init__(self, path: str, hint: str = '', cache_mode=1):
        self._key = None
        self.path = path
        self.hint = hint
        self.cache_mode = cache_mode

    def set_key(self, key: Any) -> Any:
        self._key = key
        return key

    def get_key(self) -> Any:
        if self._key is None:
            raise ValueError("scene.ExtResource key is not set")
        return self._key

    key = property(get_key, set_key)

    def load(self) -> Resource:
        return ResourceLoader.load(self.path, self.hint, self.cache_mode)


class SceneMeta(type):
    def __new__(cls, name: str, bases: tuple, attrs: Mapping[str, Any]) -> type:
        root = attrs.pop('__root_class__', _Node)
        path = attrs.pop('__path__', None)

        if path is None:
            path = 'res://' + String(name).to_snake_case().replace('_scene', '') + '.tscn'

        new_attrs = {}
        node_list = []
        ext_resource_list = []

        for name, attr in attrs.items():
            if isinstance(attr, Node):
                attr.name = name
                node_list.append(attr)
            elif isinstance(attr, ExtResource):
                attr.key = name
                ext_resource_list.append(attr)
            else:
                new_attrs[name] = attr

        new_attrs['__root_class__'] = root
        new_attrs['__path__'] = path
        new_attrs['__node_list__'] = node_list
        new_attrs['__ext_resource_list__'] = ext_resource_list

        cls = super().__new__(cls, name, bases, new_attrs)

        _scene_cache[path] = cls

        return cls


class Scene(metaclass=SceneMeta):
    __root_class__ = _Node

    def __init__(self, name=None, **kwargs):
        cls = self.__class__

        if name is None:
            if hasattr(cls, 'name'):
                name = cls.name
                name_passed = True
            else:
                name = cls.__name__
                name_passed = False

                if name.endswith('Scene'):
                    name = name[:-5]
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

        self._res = {}

        if Engine.is_editor_hint():
            # Sync scenes only when the Editor is active

            def load_resource(key: Any, path: str, hint: str = '', cache_mode=1) -> Resource:
                resource = ResourceLoader.load(path, hint, cache_mode)
                self._res[key] = resource
                return resource

            # load resources defined in the 'load' function
            if hasattr(self, 'load') and callable(self.load):
                self.load(load_resource)

            # load resources defined in the class body
            for ext_resource in cls.__ext_resource_list__:
                resource = ext_resource.load()
                self._res[ext_resource.key] = resource

            # add nodes defined in the 'create' function
            if hasattr(self, 'create') and callable(self.load):
                self.create(self._res.__getitem__)

            # add nodes defined in the class body
            for node in cls.__node_list__:
                self._add(node)

            result = packed.pack(self.root)
            if result != godot.Error.OK:
                msg = f"An error ({godot.Error(result)!r}) occurred while packing the scene ({self.root!r})."

                raise RuntimeError(msg)

            result = ResourceSaver.save(packed, self.__path__, 0)
            if result != godot.Error.OK:
                msg = f"An error ({godot.Error(result)!r}) occurred while saving the scene to disk."

                raise RuntimeError(msg)

            for value in self._res.values():
                del value

        del self._res
        del packed
        del self.root

        self._res = None
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
            class_name = attr[4:]  # attr[4:].title()
            cls = getattr(classdb, class_name)

            def add_child(name=None, **attrs):
                node = Node(cls, attrs)
                node.name = name or class_name
                self._add(node)

            return add_child

        raise AttributeError(f"{self.__class__!r} object has no attribute {attr!r}")

    def _add(self, node: Node) -> None:
        root = self.root

        if root is None:
            raise RuntimeError("Cannot add nodes after __init__")

        props = {}
        for name, prop in node.properties.items():
            if isinstance(prop, ExtResource):
                props[name] = self._res[prop.key]
            else:
                props[name] = prop

        print('Add child', node.node_class, node.name, props)
        existing = root.find_child(node.name, False, True)

        if existing:
            child = existing
            for key, value in props.items():
                child.set(key, value)
            print('Update existing', child)
        else:
            child = node.node_class(name=node.name, **props)
            root.add_child(child, False, 0)
            child.set_owner(root)
            print('Created new', child)
