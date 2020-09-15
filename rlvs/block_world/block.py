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

        self.sandbox_width = 6 * width
        self.sandbox_height = 6 * height
        self.surface_width = 2 * width
        self.surface_height = 2 * height

        
        self.width = width
        self.height = height
        self.max_score = 99999.0
        
        self._block = (mask * 3).astype('float32')
        
        max_y_shift = self.surface_height - mask.shape[1]
        max_x_shift = self.surface_width - mask.shape[0]

        self.rotate_angle = randint(-180, 180)
        self.surface_index = 1
        self.shift_x = randint(0, max_x_shift) + (self.surface_index * self.surface_width)
        self.shift_y = randint(0, max_y_shift) + (self.surface_index * self.surface_height)
        self._max_dist = self._max_distance()
        self.block_x, self.block_y = randint(1, self.sandbox_width - self._block.shape[0]), randint(1, self.sandbox_height - self._block.shape[1])

        self.update_sandbox()
        self.original_sandbox = np.copy(self.sandbox)
        self.prev_dist = None

    
    def _max_distance(self):
        bounds = np.array([
            [0, 0, -180],
            [0, self.sandbox_height, -180 ],
            [self.sandbox_width, 0, -180],
            [self.sandbox_width, self.sandbox_height, -180],
            [0, 0, 180],
            [0, self.sandbox_height, 180 ],
            [self.sandbox_width, 0, 180],
            [self.sandbox_width, self.sandbox_height, 180]
        ])

        return max(self.distance(x, y, t) for x,y,t in bounds)
        
    
    @property
    def action_bounds(self):
        return [[-8, -8, -10], [8, 8, 10]]
        #return [[-self.sandbox_width, -self.sandbox_height, -180], [self.sandbox_width, self.sandbox_height, 180]]
        
    @property
    def block(self):
        return np.round(self._block)

    @property
    def rotated_block(self):
        return np.round(ndimage.rotate(self._block, angle=-self.rotate_angle, mode='nearest',  reshape=False))

    def update_sandbox(self, translate_x = 0, translate_y = 0, rotate_angle = 0):
        mask_index = np.where(self._block == 3.0)
        shifted_mask_index = (mask_index[0] + self.shift_x,  + mask_index[1] + self.shift_y)
        
        new_x = int(np.round(translate_x)) + self.block_x
        new_y = int(np.round(translate_y)) + self.block_y

        if new_x < 0 or new_y < 0 or new_x > self.sandbox_width or new_y > self.sandbox_height:
            raise Exception('out of bounds')

        self.sandbox = np.zeros((self.sandbox_width, self.sandbox_height))
        self.sandbox[
            self.surface_index * self.surface_width:( self.surface_index + 1 ) * self.surface_width,
            self.surface_index * self.surface_height:( self.surface_index + 1 ) * self.surface_height
        ] = 1

        self.sandbox[
            self.shift_x : self.shift_x + self._block.shape[0],
            self.shift_y : self.shift_y + self._block.shape[1]
        ] += self._block
        
        self.sandbox[
            new_x : new_x + self._block.shape[0],
            new_y : new_y + self._block.shape[1]
        ] += self.rotate_block(-self.rotate_angle + rotate_angle)

        self.rotate_angle -= rotate_angle
        if self.rotate_angle < -180:
           self.rotate_angle += 360

        if self.rotate_angle > 180:
           self.rotate_angle -= 360
           
        self.block_x = new_x
        self.block_y = new_y
        
    def distance(self, x=None, y=None, theta=None):

        x = self.block_x if x is None else x
        y = self.block_y if y is None else y
        theta = self.rotate_angle if theta is None else theta

        return np.sqrt((self.shift_x - x)**2 + (self.shift_y - y )**2) + (1 - np.cos(np.deg2rad(theta/2)))
    
    def rotate_block(self, angle):
        return ndimage.rotate(self._block, angle=angle, mode='nearest',  reshape=False)

    @property
    def perfect_fit(self):
        return 0.1 >= self.distance(self.block_x, self.block_y, self.rotate_angle)
    
    def score(self):
        surface_x = range(self.surface_index * self.surface_width, ( self.surface_index + 1 ) * self.surface_width)
        surface_y = range(self.surface_index * self.surface_height, ( self.surface_index + 1 ) * self.surface_height)
        inside_surface = round(self.block_x) in surface_x and round(self.block_y) in surface_y
            
        dist = self.distance(self.block_x, self.block_y, self.rotate_angle)

        if self.prev_dist is None: # First dist
            self.prev_dist = dist

        if dist >= self.prev_dist:
            return 0 # quickly learn to avoid going out of bounds.

        score = (1 - (dist/self._max_dist)**0.4)
        
        self.prev_dist = dist
        return score
