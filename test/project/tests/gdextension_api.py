from ._base import BaseTestCase
from classes import TestResource

import gdextension

__all__ = [
    'TestCaseGDExtensionAPI'
]

class TestCaseGDExtensionAPI(BaseTestCase):
    def test_version(self):
        godot_version = gdextension.get_godot_version()
        api_version = gdextension.get_extension_api_version()
        
        self.assertEqual(godot_version.major, 4)
        self.assertEqual(godot_version[0], 4)
        self.assertEqual(api_version.major, 4)
        self.assertEqual(api_version[0], 4)

        self.assertGreaterEqual(godot_version.minor, 3)
        self.assertGreaterEqual(api_version.minor, 3)

        self.assertIsInstance(godot_version.string, str)
        self.assertIsInstance(api_version.string, str)
        self.assertEqual(godot_version.string[:16], 'Godot Engine v4.')
        self.assertEqual(api_version.string[:16], 'Godot Engine v4.')

    def test_variant_call(self):
        gdscript = self._main.get_node('TestCasesGDScript')
        m = gdextension.VariantMethod(gdscript, 'get_resource')
        # XXX: Crash in gdextension_variant_call + Variant::callp
        # resource = m()

        # self.assertIsInstance(resource, TestResource)
