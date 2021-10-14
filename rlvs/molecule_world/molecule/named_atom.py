from dataclasses import dataclass

@dataclass
class NamedAtom:
    atomic_num: int
    symbol: str
    feature_list_index: int

    def __eq__(self, other) -> bool:
        if type(other) == str:
            return self.symbol == other
        
        if type(other) == int:
            return self.atomic_num == other
        
        if hasattr(other, 'atomic_num'):
            return self.atomic_num == other.atomic_num




H = NamedAtom(1, 'H', 3)
C = NamedAtom(6, 'C', 5)
N = NamedAtom(7, 'N', 6)
O = NamedAtom(8, 'O', 7)
S = NamedAtom(16, 'S', 9)

## Halogens

F = NamedAtom(9, 'F', 11)
Cl = NamedAtom(17, 'Cl', 11)
Br = NamedAtom(35, 'Br', 11)
I = NamedAtom(53, 'I', 11)
