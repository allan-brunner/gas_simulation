import os
import h5py
import numpy as np

from gas_simulation.atom import Atom
from gas_simulation.domain import Domain

class Simulation:
    def __init__(self, atoms: np.ndarray | list[Atom], domain: Domain):
        self.atoms = atoms
        self.h5_file: h5py.File | None = None
        self.xdmf_file = None
        self.h5_filename: str = ""
        self.step_count: int = 0

        self.domain = domain

    def simulate(self, t):
        for atom in self.atoms:
            atom.move(t, self.domain)

        pos = np.array([atom.spatial_pos for atom in self.atoms])
        radii = np.array([atom.radius for atom in self.atoms])

        diff = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        min_dist2 = (radii[:, None] + radii[None, :]) ** 2

        i_indices, j_indices = np.where(np.triu(dist2 <= min_dist2, k=1))

        for i,j in zip(i_indices, j_indices):
            self.atoms[i].transfer_speed(self.atoms[j])

    def open_export(self, filename: str = "simulation", folder: str = "export"):
        """
        Crée le dossier d'exportation, initialise les fichiers .h5 et .xdmf,
        et écrit l'en-tête XML ainsi que le maillage de contour (boundary).
        """
        # Création du dossier d'export si nécessaire
        if folder:
            os.makedirs(folder, exist_ok=True)
            h5_path = os.path.join(folder, f"{filename}.h5")
            xdmf_path = os.path.join(folder, f"{filename}.xdmf")
        else:
            h5_path = f"{filename}.h5"
            xdmf_path = f"{filename}.xdmf"

        self.h5_filename = os.path.basename(h5_path)
        self.step_count = 0

        # Ouverture du fichier HDF5
        self.h5_file = h5py.File(h5_path, "w")

        # Ouverture du fichier XDMF et écriture de l'en-tête XML
        self.xdmf_file = open(xdmf_path, "w", encoding="utf-8")
        header = f"""<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [] >
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="boundary" GridType="Uniform">
            <Topology TopologyType="Hexahedron" NumberOfElements="1">
                <DataItem Dimensions="1 8" NumberType="Int" Format="XML">
                    0 1 2 3 4 5 6 7
                </DataItem>
            </Topology>
            <Geometry GeometryType="XYZ">
                <DataItem Dimensions="8 3" NumberType="Float" Precision="8" Format="XML">
                    {-self.domain.lx/2} {-self.domain.ly/2} {-self.domain.lz/2} {self.domain.lx/2} {-self.domain.ly/2} {-self.domain.lz/2} {self.domain.lx/2} {self.domain.ly/2} {-self.domain.lz/2} {-self.domain.lx/2} {self.domain.ly/2} {-self.domain.lz/2} {-self.domain.lx/2} {-self.domain.ly/2} {self.domain.lz/2} {self.domain.lx/2} {-self.domain.ly/2} {self.domain.lz/2} {self.domain.lx/2} {self.domain.ly/2} {self.domain.lz/2} {-self.domain.lx/2} {self.domain.ly/2} {self.domain.lz/2}
                </DataItem>
            </Geometry>
        </Grid>
        <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
"""
        self.xdmf_file.write(header)
        self.xdmf_file.flush()

    def write_step_into_export(self, time: float = 0.0):
        """
        Sauvegarde le pas de temps courant dans le fichier .h5 (positions, vitesses,
        masses, rayons) et référence ces datasets dans le fichier .xdmf.
        """
        if self.h5_file is None or self.xdmf_file is None:
            raise RuntimeError("L'export n'a pas été ouvert. Appelez open_export d'abord.")

        step_name = f"step_{self.step_count:04d}"
        n_atoms = len(self.atoms)

        # Extraction des données sous forme de tableaux NumPy
        positions = np.array([atom.spatial_pos for atom in self.atoms], dtype=np.float64)
        velocities = np.array([atom.speed for atom in self.atoms], dtype=np.float64)
        masses = np.array([atom.mass for atom in self.atoms], dtype=np.float64)
        radii = np.array([atom.radius for atom in self.atoms], dtype=np.float64)

        # Écriture dans le fichier HDF5
        grp = self.h5_file.create_group(step_name)
        grp.create_dataset("positions", data=positions)
        grp.create_dataset("velocity", data=velocities)
        grp.create_dataset("mass", data=masses)
        grp.create_dataset("radius", data=radii)
        self.h5_file.flush()

        # Ajout du bloc Grid dans le fichier XDMF
        grid_xml = f"""            <Grid Name="{step_name}" GridType="Uniform">
                <Time Value="{time}"/>
                <Topology TopologyType="Polyvertex" NumberOfElements="{n_atoms}"/>
                <Geometry GeometryType="XYZ">
                    <DataItem Dimensions="{n_atoms} 3" NumberType="Float" Precision="8" Format="HDF">{self.h5_filename}:/{step_name}/positions</DataItem>
                </Geometry>
                <Attribute Name="mass" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="{n_atoms}" NumberType="Float" Precision="8" Format="HDF">{self.h5_filename}:/{step_name}/mass</DataItem>
                </Attribute>
                <Attribute Name="radius" AttributeType="Scalar" Center="Node">
                    <DataItem Dimensions="{n_atoms}" NumberType="Float" Precision="8" Format="HDF">{self.h5_filename}:/{step_name}/radius</DataItem>
                </Attribute>
                <Attribute Name="velocity" AttributeType="Vector" Center="Node">
                    <DataItem Dimensions="{n_atoms} 3" NumberType="Float" Precision="8" Format="HDF">{self.h5_filename}:/{step_name}/velocity</DataItem>
                </Attribute>
            </Grid>
"""
        self.xdmf_file.write(grid_xml)
        self.xdmf_file.flush()

        self.step_count += 1

    def close_export(self):
        """
        Referme les balises XML ouvertes et ferme les fichiers .h5 et .xdmf.
        """
        if self.xdmf_file is not None and not self.xdmf_file.closed:
            self.xdmf_file.write("        </Grid>\n")
            self.xdmf_file.write("    </Domain>\n")
            self.xdmf_file.write("</Xdmf>\n")
            self.xdmf_file.close()

        if self.h5_file is not None:
            self.h5_file.close()

