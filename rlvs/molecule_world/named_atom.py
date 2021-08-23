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



