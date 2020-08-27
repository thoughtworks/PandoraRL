from random import randint
import numpy as np
class Block:
    masks = [
        np.array(
            [[1, 1, 1, 1],
             [0, 1, 1, 0],
             [0, 1, 1, 0],
             [0, 1, 1, 0]]
        ),

        np.array(
            [[1, 1, 1, 1],
             [1, 1, 1, 0],
             [1, 1, 0, 0],
             [1, 0, 0, 0]]
        ),

        
        np.array(
            [[1, 1, 1],
             [1, 1, 0],
             [1, 0, 0],
             [1, 0, 0]]
        ),

        np.array(
            [[1, 1, 1, 1],
             [0, 0, 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0]]
        ),

        np.array(
            [[1],
             [1],
             [1],
             [1]]
        ),

        np.array(
            [[1, 1, 1, 1]]
        ),

        np.array(
            [[1, 1, 1, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]]
        )
    ]

    @classmethod
    def get_block(cls, width=10, height=10):
        mask = cls.masks[randint(0, len(cls.masks) - 1)]
        return Block(width, height, mask)


    def __init__(self, width, height, mask):
        template = np.ones((width, height), dtype=int)
        max_shift = template.shape[1] - mask.shape[1]
        shift = randint(0, max_shift)

        mask_index = np.where(mask == 1)
        shifted_mask_index = (mask_index[0], mask_index[1] + shift)
        template[shifted_mask_index] = 0
        self.surface = template
        self.block = mask
