from random import randint
from scipy import ndimage
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
        template = np.ones((width, height)).astype('float64')
        max_y_shift = template.shape[1] - mask.shape[1]
        max_x_shift = template.shape[0] - mask.shape[0]
        self.shift_x = randint(0, max_x_shift)
        self.shift_y = randint(0, max_y_shift)

        mask_index = np.where(mask == 1)
        shifted_mask_index = (mask_index[0] + self.shift_x, mask_index[1] + self.shift_y)
        template[shifted_mask_index] = np.NaN
        self.surface = template
        self._rotate_angle = randint(-180, 180)
        self._block = ndimage.rotate(mask.astype('float64'), angle=self._rotate_angle, mode='nearest',  reshape=False)
        self.sandbox = np.zeros((2*width + 1, 2*height + 1))
        self.sandbox[width:2*width, height:2*height] = self.surface
        self.sandbox[1:self._block.shape[0] + 1, 1:self._block.shape[1] + 1] = self._block

    @property
    def block(self):
        return np.round(self._block)

    @property
    def rotated_block(self):
        return np.round(ndimage.rotate(self._block, angle=-self._rotate_angle, mode='nearest',  reshape=False))

    def rotate_block(self, angle):
        return np.round(ndimage.rotate(self._block, angle=-angle, mode='nearest',  reshape=False))

