import unittest

from gas_simulation.domain import Domain

class TestDomain(unittest.TestCase):
    def test_volume(self):
        self.assertEqual(Domain(0, 0, 0).volume(), 0)
        self.assertEqual(Domain(1, 2, 3).volume(), 6)
        self.assertEqual(Domain(3, 3, 3).volume(), 27)
