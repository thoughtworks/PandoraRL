import json

class Config:
    __instance = None

    def __init__(self, config_path):

        # pre initialize to avoid errors
        self.single_step = [1]
        self.node_features = [
            'atom_type_encoding',
            'atom_named_features',
            'is_heavy_atom',
            'VDWr',
            'molecule_type',
            'smarts_patterns',
            'residue',
            'z_scores',
            'kd_hydophobocitya',
            'conformational_similarity'
        ]

        self.edge_features = [
            "encoding",
            "bond_distance"
        ]

        self.test_dataset = [
            'SARSTest', 'SARSCov2'
        ]

        self.train_dataset = [
            'SARSVarients', 'MERSVariants'
        ]
        
        config = {}
        with open(config_path, 'r') as config_f:
            config = json.load(config_f)

        self.__dict__.update(config)

    @classmethod
    def init(cls, config_path):
        cls.__instance = Config(config_path)

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            raise Exception('Instance not initialized, use Config.init to initialize the config')

        return cls.__instance
