class DTree:
    __slots__ = 'node_name', 'true_branch', 'false_branch', 'total_targets', 'target_set', 'attributes', 'depth'

    def __init__(self, node_name, total_targets, target_set, attributes, depth):
        self.node_name = node_name
        self.true_branch = None
        self.false_branch = None
        self.total_targets = total_targets
        self.target_set = target_set
        self.attributes = attributes
        self.depth = depth