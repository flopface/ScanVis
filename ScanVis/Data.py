from .Scan import *
from matplotlib.colors import LinearSegmentedColormap

class Data(Scan):
  def __init__(self, array = None, parent : Scan = None):
    if type(array) != type(None):
      self.array = array * parent.mask_array + (~(parent.mask_array.astype(bool))).astype(int)
      biggest = np.max(self.array)
      smallest = np.min(self.array)
      if biggest-1 > 1-smallest: smallest = 2-biggest
      else: biggest = 2-smallest
      self.array[:,0,0] = smallest
      self.array[0,:,0] = smallest
      self.array[0,0,:] = smallest
      self.array[:,1,1] = biggest
      self.array[1,:,1] = biggest
      self.array[1,1,:] = biggest

    self.scan_file = parent.scan_file
    self.seg_file = parent.seg_file
    self.seg_array = parent.seg_array
    self.id = parent.id
    self.age = parent.age
    self.gender = parent.gender
    self.spacing = parent.spacing
    self.direction = parent.direction
    self.origin = parent.origin
    self.voxel_volume = parent.voxel_volume
    self.shape = parent.shape
    self.volume = parent.volume
    self.cmap = LinearSegmentedColormap.from_list("blue_black_red", [(0, 0, 1), (0, 0, 0), (1, 0, 0)], N=256)
  
  @property
  def scan_array(self): return self.array
  @property
  def shifted_seg_array(self): return self.array
  @property
  def mask_array(self): return self.array
  @property
  def brain_array(self): return self.array