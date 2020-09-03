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
        self.surface = np.ones((width, height)).astype('float64')
        self.width = width
        self.height = height
        self.max_score = 99999.0
        
        self._block = (mask * 3).astype('float64')
        max_y_shift = self.surface.shape[1] - mask.shape[1]
        max_x_shift = self.surface.shape[0] - mask.shape[0]


        self.rotate_angle = randint(-180, 180)
        self.surface_index = np.random.randint(1, 3)
        self.shift_x = randint(0, max_x_shift) + (self.surface_index * self.width)
        self.shift_y = randint(0, max_y_shift) + (self.surface_index * self.height)

        block_index = 3 - self.surface_index

        self.update_sandbox(block_index * self.width, block_index * self.height)
        self.original_sandbox = np.copy(self.sandbox)
        
    @property
    def block(self):
        return np.round(self._block)

    @property
    def rotated_block(self):
        return np.round(ndimage.rotate(self._block, angle=-self.rotate_angle, mode='nearest',  reshape=False))

    def update_sandbox(self, translate_x = 0, translate_y = 0, rotate_angle = 0):
        mask_index = np.where(self._block == 3.0)
        shifted_mask_index = (mask_index[0] + self.shift_x,  + mask_index[1] + self.shift_y)
        
        new_x = int(np.round(np.abs(translate_x)))
        new_y = int(np.round(np.abs(translate_y)))

        self.sandbox = np.zeros((4 * self.width, 4 * self.height))
        self.sandbox[
            self.surface_index * self.width:( self.surface_index + 1 ) * self.width,
            self.surface_index * self.height:( self.surface_index + 1 ) * self.height
        ] = 1
        self.sandbox[
            self.shift_x : self.shift_x + self._block.shape[0],
            self.shift_y : self.shift_y + self._block.shape[1]
        ] += self._block
        
        self.sandbox[
            new_x : new_x + self._block.shape[0],
            new_y : new_y + self._block.shape[1]
        ] += self.rotate_block(-self.rotate_angle + rotate_angle)

        
    
    def rotate_block(self, angle):
        return ndimage.rotate(self._block, angle=angle, mode='nearest',  reshape=False)

    def score(self, shift_x, shift_y, rotate_angle):
        pi_2 = np.deg2rad(90)
        r = np.sqrt(shift_x**2 + shift_y**2)
        phi =  pi_2 if shift_x == 0 else np.arctan(shift_y/shift_x)
        theta = phi + np.deg2rad(rotate_angle)

        r_actual = np.sqrt(self.shift_x**2 + self.shift_y**2)
        phi_actual = pi_2 if self.shift_x == 0 else np.arctan(self.shift_y/self.shift_x)
        theta_actual = phi_actual + np.deg2rad(self.rotate_angle)

        dist = np.sqrt(r**2 + r_actual**2 - 2*(r * r_actual * np.cos(theta - theta_actual)))

        return self.max_score if dist == 0 else 1/dist
