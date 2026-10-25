import numpy as np

from gas_simulation.domain import Domain

class Atom:

    # Atomic mass in kg
    # Atomic radius in m
    # Chemical formula
    # Spatial position for atom center in (x, y, z)
    # Speed in (x, y, z)/s
    def __init__(self, mass, radius, formula, spatial_pos: np.ndarray, speed: np.ndarray) -> None:
        self.mass = mass
        self.radius = radius
        self.formula = formula
        self.spatial_pos = np.array(spatial_pos, dtype=np.float64)
        self.speed = np.array(speed, dtype=np.float64)

    def move(self, t, domain: Domain):
        newX = self.spatial_pos[0] + self.speed[0] * t
        newY = self.spatial_pos[1] + self.speed[1] * t
        newZ = self.spatial_pos[2] + self.speed[2] * t

        self.check_bounds(newX, newY, newZ, domain)

    def check_bounds(self, newX, newY, newZ, domain: Domain):
        xboundneg = newX - self.radius
        xboundpos = newX + self.radius
        if xboundneg <= -domain.lx / 2:
            newX += 2 * (-domain.lx / 2 - xboundneg)
            self.speed[0] *= -1
        if xboundpos > domain.lx / 2:
            newX += 2 * (domain.lx / 2 - xboundpos)
            self.speed[0] *= -1

        yboundneg = newY - self.radius
        yboundpos = newY + self.radius
        if yboundneg <= -domain.ly / 2:
            newY += 2 * (-domain.ly / 2 - yboundneg)
            self.speed[1] *= -1
        if yboundpos > domain.ly / 2:
            newY += 2 * (domain.ly / 2 - yboundpos)
            self.speed[1] *= -1

        zboundneg = newZ - self.radius
        zboundpos = newZ + self.radius
        if zboundneg <= -domain.lz / 2:
            newZ += 2 * (-domain.lz / 2 - zboundneg)
            self.speed[2] *= -1
        if zboundpos > domain.lz / 2:
            newZ += 2 * (domain.lz / 2 - zboundpos)
            self.speed[2] *= -1

        self.spatial_pos[0] = newX
        self.spatial_pos[1] = newY
        self.spatial_pos[2] = newZ

    def check_collision(self, other: Atom):
        delta_x = self.spatial_pos[0] - other.spatial_pos[0]
        delta_y = self.spatial_pos[1] - other.spatial_pos[1]
        delta_z = self.spatial_pos[2] - other.spatial_pos[2]
        dist2 = delta_x ** 2 + delta_y ** 2 + delta_z ** 2

        return dist2 <= (self.radius + other.radius) ** 2

    def transfer_speed(self, other: Atom):
        sum_mass = self.mass + other.mass
        delta_vij = self.speed - other.speed
        delta_rij = self.speed - other.speed

        dist2 = float(np.dot(delta_rij, delta_rij))
        if dist2 == 0.0:
            return

        proj = (np.dot(delta_vij, delta_rij) / dist2) * delta_rij

        self.speed = self.speed - (2 * other.mass / sum_mass) * proj
        other.speed = other.speed + (2 * self.mass / sum_mass) * proj

    def __str__(self) -> str:
        return "Atom(mass: " + str(self.mass) + ", radius: " + str(self.radius) + ", formula: " + str(self.formula) + ", spatial_pos: " + str(self.spatial_pos) + ", speed: " + str(self.speed) + ")"

