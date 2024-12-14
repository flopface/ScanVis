from __future__ import annotations
from .Scan import *
import ants

class Scant(Scan):
  def __init__(self, scan_file, seg_file, transform_folder = None, age = 0, gender = 'Unknown', cmap = 'inferno'):
    super().__init__(scan_file, seg_file)

  def array2ants(self, array):
    ants_image = ants.from_numpy(array)
    ants_image.set_spacing(self.spacing)
    ants_image.set_origin(self.origin)
    ants_image.set_direction(self.direction)
    return ants_image
  
  def transform(self, transform, fix : Scant):
    fix = fix.array2ants(fix.arrays['scan'])
    self.arrays['scan'] = ants.apply_transforms(fixed=fix, moving=self.array2ants(self.arrays['scan']), transformlist=transform,  interpolator='nearestNeighbor').numpy()
    self.arrays['seg'] = ants.apply_transforms(fixed=fix, moving=self.array2ants(self.arrays['seg']), transformlist=transform,  interpolator='nearestNeighbor').numpy()
    self.arrays['shifted_seg'] = self.arrays['seg'] + 15*(self.arrays['seg'].astype(bool).astype(int))
    self.arrays['mask'] = self.arrays['seg'].astype(bool)
    self.arrays['brain'] = self.arrays['scan'] * self.arrays['mask']
    self.normalise_color()

  def register(self, other : Scan, return_result = False):
    fix = other.array2ants(other.arrays['brain'])
    result = ants.registration(
      fixed = fix,
      moving = self.array2ants(self.arrays['brain']),
      type_of_transform = 'SyN',
      aff_shrink_factors=(6, 4, 2, 1),
      aff_smoothing_sigmas=(3, 2, 1, 0)
    )
    if return_result: return result
    self.transform(result['fwdtransforms'], other)
