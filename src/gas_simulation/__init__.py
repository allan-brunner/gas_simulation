import math
import numpy as np
from random import random
import time

from gas_simulation.atom import Atom
from gas_simulation.simulation import Simulation
from gas_simulation.domain import Domain

def main():
    start = time.time()
    atoms = []

    domain = Domain(5e-9, 5e-9, 5e-9)
    atom_he_radius = 1.1e-10
    atom_he_mass = 6.646e-27
    nb_atoms = 400
    # For 400, should be 8
    ceil_atom_per_row = math.ceil(nb_atoms ** (1/3))
    space_x = 2 * domain.lx / ceil_atom_per_row
    space_y = 2 * domain.ly / ceil_atom_per_row
    space_z = 2 * domain.lz / ceil_atom_per_row

    def init_atoms():
        print("Initiating atoms")
        for i in range(ceil_atom_per_row):
            pos_x = (-domain.lx) / 2 + (space_x / 2) * i
            for j in range(ceil_atom_per_row):
                pos_y = (-domain.ly) / 2 + (space_y / 2) * j
                for k in range(ceil_atom_per_row):
                    if len(atoms) >= 400:
                        return
                    pos_z = (-domain.lz) / 2 + (space_z / 2) * k
       
                    inital_speed = 1400
                    phi = random() * 2 * math.pi
                    cos_theta = 2 * random() - 1
                    sin_theta = math.sqrt(1 - cos_theta ** 2)

                    vx = inital_speed * sin_theta * math.cos(phi)
                    vy = inital_speed * sin_theta * math.sin(phi)
                    vz = inital_speed * cos_theta

                    atoms.append(Atom(atom_he_mass, atom_he_radius, "He", np.array([pos_x, pos_y, pos_z]), np.array([vx, vy, vz])))

    init_atoms()

    sim = Simulation(atoms, domain)

    sim.open_export(filename="simulation", folder="export")

    finalT = 2e-11
    deltaT = 1e-14
    t = 0
    total_steps = int(finalT / deltaT)
    for i in range(total_steps):
        sim.simulate(deltaT)
        t += deltaT
        sim.write_step_into_export(time=t)
        if i % 250 == 0:
            print(f"Simulated frame {i} / {total_steps}")

    sim.close_export()

    end = time.time() - start

    print("Export terminé avec succès dans le dossier 'export/' !")
    print(f"Took {end:.2f}s")
