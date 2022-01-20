class InterMolecularInteractions:
    def __init__(self):
        self.gaussian = 0
        self.repulsion = 0
        self.hydrophobic = 0
        self.hydrogen = 0

    def update_interaction_strength(self, bond):
        self.gaussian += bond.gauss1 + bond.gauss2
        self.repulsion += bond.repulsion
        self.hydrogen += bond.hydrogenbond
        self.hydrophobic += bond.hydrophobic

    def reset_interaction_strengths(self):
        self.gaussian = 0
        self.repulsion = 0
        self.hydrophobic = 0
        self.hydrogen = 0

    @property
    def features(self):
        return [self.gaussian, self.repulsion, self.hydrophobic, self.hydrogen]
