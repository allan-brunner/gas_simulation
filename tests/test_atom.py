import numpy as np
import unittest

from gas_simulation.atom import Atom
from gas_simulation.domain import Domain

class TestAtom(unittest.TestCase):

    def test_atom_move(self):
        dt = 1
        domain = Domain(6, 6, 6)
        atom_x = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([1, 0, 0]))
        atom_y = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([0, 1, 0]))
        atom_z = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([0, 0, 1]))

        atom_x.move(dt, domain)
        self.assertEqual(atom_x.spatial_pos[0], 2)

        atom_y.move(dt, domain)
        self.assertEqual(atom_y.spatial_pos[1], 2)
        
        atom_z.move(dt, domain)
        self.assertEqual(atom_z.spatial_pos[2], 2)

    def test_atom_reflectivity(self):
        dt = 1
        domain = Domain(6, 6, 6)
        atom_x = Atom(1, 1, "He", np.array([2, 1, 1]), np.array([1, 0, 0]))
        atom_y = Atom(1, 1, "He", np.array([1, 2, 1]), np.array([0, 1, 0]))
        atom_z = Atom(1, 1, "He", np.array([1, 1, 2]), np.array([0, 0, 1]))

        atom_x.move(dt, domain)
        self.assertEqual(atom_x.spatial_pos[0], 1)

        atom_y.move(dt, domain)
        self.assertEqual(atom_y.spatial_pos[1], 1)
        
        atom_z.move(dt, domain)
        self.assertEqual(atom_z.spatial_pos[2], 1)

    def test_atom_collision_detection(self):
        atom_1 = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([0, 0, 0]))
        atom_2 = Atom(2, 1, "He", np.array([3, 1, 1]), np.array([-1, 0, 0]))

        self.assertTrue(atom_1.check_collision(atom_2))

    def test_atom_no_collision(self):
        atom_1 = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([0, 0, 0]))
        atom_2 = Atom(1, 1, "He", np.array([5, 1, 1]), np.array([-1, 0, 0]))

        self.assertFalse(atom_1.check_collision(atom_2))

    def test_atom_collision_same_mass(self):
        dt = 1
        domain = Domain(10, 10, 10)
        atom_1 = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([0, 0, 0]))
        atom_2 = Atom(1, 1, "He", np.array([3, 1, 1]), np.array([-1, 0, 0]))
        atom_1_speed = atom_1.speed.copy()
        atom_2_speed = atom_2.speed.copy()

        atom_1.move(dt, domain)
        atom_2.move(dt, domain)
        if atom_1.check_collision(atom_2):
            atom_1.transfer_speed(atom_2)
        
        for i in range(3):
            self.assertAlmostEqual(atom_1.speed[i], atom_2_speed[i])
            self.assertAlmostEqual(atom_2.speed[i], atom_1_speed[i])

    def test_atom_collision_diff_mass_a1_no_speed(self):
        atom_1 = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([0, 0, 0]))
        atom_2 = Atom(2, 1, "He", np.array([3, 1, 1]), np.array([-1, 0, 0]))
        atom_2_speed = atom_2.speed.copy()

        if atom_1.check_collision(atom_2):
            atom_1.transfer_speed(atom_2)

        for i in range(3):
            self.assertAlmostEqual(atom_1.speed[i], 4/3 * atom_2_speed[i])
            self.assertAlmostEqual(atom_2.speed[i], 1/3 * atom_2_speed[i])

    def test_atom_collision_diff_mass_a2_no_speed(self):
        atom_1 = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([1, 0, 0]))
        atom_2 = Atom(2, 1, "He", np.array([3, 1, 1]), np.array([0, 0, 0]))
        atom_1_speed = atom_1.speed.copy()

        if atom_1.check_collision(atom_2):
            atom_1.transfer_speed(atom_2)

        for i in range(3):
            self.assertAlmostEqual(atom_1.speed[i], -1/3 * atom_1_speed[i])
            self.assertAlmostEqual(atom_2.speed[i], 2/3 * atom_1_speed[i])

    def test_atom_collision_diff_mass(self):
        atom_1 = Atom(1, 1, "He", np.array([1, 1, 1]), np.array([1, 0, 0]))
        atom_2 = Atom(2, 1, "He", np.array([3, 1, 1]), np.array([-1, 0, 0]))
        atom_1_speed = atom_1.speed.copy()
        atom_2_speed = atom_2.speed.copy()

        if atom_1.check_collision(atom_2):
            atom_1.transfer_speed(atom_2)

        for i in range(3):
            self.assertAlmostEqual(atom_1.speed[i], atom_1_speed[i] * -1/3 + 4/3 * atom_2_speed[i])
            self.assertAlmostEqual(atom_2.speed[i], atom_1_speed[i] * 2/3 +  1/3 * atom_2_speed[i])

if __name__ == '__main__':
    unittest.main()
