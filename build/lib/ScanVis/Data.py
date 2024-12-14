from .Scant import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class Data(Scant):
  def __init__(self, array, parent : Scan):
    self.array = np.flip(array, 1)
    self.arrays = dict()

    self.arrays['mask'] = parent.arrays['mask']
    self.arrays['seg'] = parent.arrays['seg']
    self.arrays['shifted_seg'] = parent.arrays['shifted_seg']
    self.arrays['scan'] = self.array
    self.arrays['scan'] = self.arrays['scan'] + (~(self.arrays['scan'].astype(bool))).astype(int)
    self.arrays['brain'] = self.array * parent.arrays['mask'] + (~(parent.arrays['mask'].astype(bool))).astype(int)
    
    self.normalise_color()

    self.scan_file = parent.scan_file
    self.seg_file = parent.seg_file
    self.seg_array = parent.arrays['seg']
    self.id = parent.id
    self.age = parent.age
    self.gender = parent.gender
    self.spacing = parent.spacing
    self.direction = parent.direction
    self.origin = parent.origin
    self.voxel_volume = parent.voxel_volume
    self.shape = parent.arrays['scan'].shape
    self.volume = parent.volume
    colors = [
    (1, 1, 1),  # Bright blue
    (0, 1, 1),  # Bright blue
    (0, 0, 1),  # Bright blue
    (0, 0, 0),  # Black (center)
    (1, 0, 0),  # Bright red
    (1, 1, 0),  # Bright red
    (1, 1, 1)   # Bright red
    ]
    self.cmap = LinearSegmentedColormap.from_list("blue_black_red_adjusted", colors, N=256)

    #self.cmap = LinearSegmentedColormap.from_list("blue_black_red", [(0, 0, 1), (0, 0, 0), (1, 0, 0)], N=256)    
  
  def normalise_color(self):
    for key in ['shifted_seg', 'seg', 'scan', 'brain']:
      biggest = np.max(self.arrays[key])
      smallest = np.min(self.arrays[key])
      if key in ['scan', 'brain']:
        if biggest > -smallest: smallest = (2)-biggest
        else: biggest = (2)-smallest
      self.arrays[key][:,0,0] = smallest
      self.arrays[key][0,:,0] = smallest
      self.arrays[key][0,0,:] = smallest
      self.arrays[key][:,1,1] = biggest
      self.arrays[key][1,:,1] = biggest
      self.arrays[key][1,1,:] = biggest

  def transform(self, transform, fix : Scant):
    fix = fix.array2ants(fix.arrays['scan'])
    self.arrays['scan'] = ants.apply_transforms(fixed=fix, moving=self.array2ants(self.arrays['scan']), transformlist=transform,  interpolator='nearestNeighbor').numpy()
    self.arrays['seg'] = ants.apply_transforms(fixed=fix, moving=self.array2ants(self.arrays['seg']), transformlist=transform,  interpolator='nearestNeighbor').numpy()
    self.arrays['shifted_seg'] = self.arrays['seg'] + 15*(self.arrays['seg'].astype(bool).astype(int))
    self.arrays['mask'] = self.arrays['seg'].astype(bool)
    self.arrays['brain'] = self.arrays['scan'] * self.arrays['mask'] + (~(self.arrays['mask'].astype(bool))).astype(int)
    self.normalise_color()

  def get_vals(self, id):
    if id: truth = ~(self.arrays['seg'] - id).astype(bool)
    else: truth = (self.arrays['seg'] - id).astype(bool)
    return self.arrays['scan'][np.where(truth)]
