class Domain:
    def __init__(self, lx: float, ly: float, lz: float):
        self.lx = lx
        self.ly = ly
        self.lz = lz

    def volume(self):
        return self.lx * self.ly * self.lz

