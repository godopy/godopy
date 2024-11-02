import numpy as np

from godot.types import *

from ._base import BaseTestCase


__all__ = [
    'TestCaseAtomicTypes',
    'TestCaseMathTypes'
]


class TestCaseAtomicTypes(BaseTestCase):
    def test_string(self):
        s = String('GodoPy')
        self.assertIsInstance(s, str)
        self.assertIsInstance(s, String)

        self.assertEqual(s, 'GodoPy')

        s2 = s.to_snake_case()
        self.assertEqual(s2, 'godo_py')

        self.assertEqual(s2.to_camel_case(), 'godoPy')
        self.assertEqual(s2.to_pascal_case(), 'GodoPy')
        self.assertEqual(s2.capitalize(), 'Godo_py')  # native Python str's method
        self.assertEqual(s2.upper(), s2.to_upper())
        self.assertEqual(s2.lower(), s2.to_lower())
        self.assertEqual(s2.is_empty(), False)
        self.assertEqual(String().is_empty(), True)
        self.assertEqual(s.path_join('tests').path_join('string'), 'GodoPy/tests/string')


class TestCaseMathTypes(BaseTestCase):
    def test_vector2(self):
        v = Vector2(2.5, 5)
        self.assertEqual(v.dtype, np.dtype('float32'))
        self.assertIsInstance(v[0], np.float32)
        self.assertIsInstance(v.x, np.float32)
        self.assertEqual(list(v), [2.5, 5.])

        v.x = 10.0
        self.assertEqual(list(v), [10., 5.])

        v.coord = (2, 1)
        self.assertEqual(list(v), [2., 1.])

        v2 = Vector2i(5, 10)
        self.assertIsInstance(v2, Vector2i)
        self.assertEqual(v2.dtype, np.dtype('int32'))
        self.assertIsInstance(v2.x, np.int32)
        self.assertEqual(list(v2), [5, 10])

        v3 = as_vector2(v2)
        self.assertIsInstance(v3, Vector2)
        self.assertEqual(v3.dtype, np.dtype('float32'))
        self.assertEqual(list(v3), [5., 10.])

        v4 = as_vector2(v3, dtype=np.float64)
        self.assertIsInstance(v4, Vector2)
        self.assertEqual(v4.dtype, np.dtype('float64'))

        v5 = as_vector2i(v4, dtype=np.int8)
        self.assertIsInstance(v5, Vector2i)
        self.assertEqual(v5.dtype, np.dtype('int8'))

    def test_rect(self):
        r = Rect2(0, 0, 100, 200)
        self.assertIsInstance(r, Rect2)
        self.assertEqual(r.dtype, np.dtype('float32'))
        self.assertIsInstance(r.position, Vector2)
        self.assertIsInstance(r.position.x, np.float32)
        self.assertIsInstance(r.size_, Size2)
        self.assertEqual(list(r), [0., 0., 100., 200.])

        r.position = (2, 5)
        self.assertEqual(list(r), [2., 5., 100., 200.])

        r.position.x = 10
        self.assertEqual(list(r), [10., 5., 100., 200.])

        r.size_.height = 50
        self.assertEqual(list(r), [10., 5., 100., 50.])

        self.assertEqual(list(r.position), [10., 5.])
        self.assertEqual(list(r.size_), [100., 50.])

        r2 = Rect2i(0, 0, 100, 200)
        self.assertIsInstance(r2, Rect2i)
        self.assertEqual(r2.dtype, np.dtype('int32'))
        self.assertIsInstance(r2.position, Vector2i)
        self.assertIsInstance(r2.position.x, np.int32)
        self.assertEqual(list(r2), [0, 0, 100, 200])

        # Type casting and reshaping
        r3 = as_rect2(np.array([[10, 20], [100, 80]]))
        self.assertEqual(r.dtype, np.dtype('float32'))
        self.assertEqual(list(r3), [10, 20, 100, 80])
