import json


class Config:
    __instance = None

    def __init__(self, config_path):

        # pre initialize to avoid errors
        self.single_step = [1, 1, 1, 1, 1, 1]
        self.node_features = [
            'atom_type_encoding',
            'atom_named_features',
            'is_heavy_atom',
            'VDWr',
            'molecule_type',
            'smarts_patterns',
            'residue_labels',
            'z_scores',
            'kd_hydophobocitya',
            'conformational_similarity'
        ]

        # comment out the "encoding" and "bond distance" features if using binding affinity
        # (since double information)

        self.edge_features = [
           "encoding",
           "bond_distance",
        ]

        # Use either: "binding affinity (sum over all affinities using Vinascore weights)
        # or: individual bond information for e.g. "hydrogenbond", "repulsion"

        # self.edge_features = [
        #     "bind_aff"
        # ]

        # self.edge_features = [
        #     "g1",
        #     "g2",
        #     "rep",
        #     "hyph",
        #     "hydr"
        # ]

        self.test_dataset = [
            'SARSTest', 'SARSCov2'
        ]

        self.train_dataset = [
            'SARSVarients', 'MERSVariants'
        ]

        self.run_tests = True
        self.test_from_episode = 70
        self.divergence_slope = 0.005

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
