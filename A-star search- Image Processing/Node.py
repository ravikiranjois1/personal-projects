class Node:
    __slots__ = 'x_coord', 'y_coord', 'parent_node', 'time_taken'

    def __init__(self, x_coord, y_coord, parent_node=None, time_taken = 0):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.parent_node = parent_node
        self.time_taken = time_taken

    def __eq__(self, other):
        self.x_coord == other.x_coord and self.y_coord == other.y_coord