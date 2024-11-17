from __future__ import annotations
from .Scan import *
from .Data import *
import ants

class Scant(Scan):
  def __init__(self, scan_file, seg_file = None, scan_folder = '/Users/work/Desktop/MPhys/popty-ping/scans', seg_folder = '/Users/work/Desktop/MPhys/popty-ping/segmentations', transform_folder = '/Users/work/Desktop/MPhys/transforms'):
    super().__init__(scan_file, seg_file, scan_folder, seg_folder)

  def array2ants(self, array):
    ants_image = ants.from_numpy(array)
    ants_image.set_spacing(self.spacing)
    ants_image.set_origin(self.origin)
    ants_image.set_direction(self.direction)
    return ants_image
  
  def transform(self, transform, fix):
    self.scan_array = ants.apply_transforms(fixed=fix, moving=self.array2ants(self.scan_array), transformlist=transform,  interpolator='nearestNeighbor').numpy()
    self.seg_array = ants.apply_transforms(fixed=fix, moving=self.array2ants(self.seg_array), transformlist=transform,  interpolator='nearestNeighbor').numpy()
    self.shifted_seg_array = self.seg_array + 15*(self.seg_array.astype(bool).astype(int))
    self.mask_array = self.seg_array.astype(bool)
    self.brain_array = self.scan_array * self.mask_array

  def register(self, other : Scant, jacobian = False, return_result = False):
    fix = other.array2ants(other.brain_array)
    result = ants.registration(
      fixed = fix,
      moving = self.array2ants(self.brain_array),
      type_of_transform = 'SyN',
      aff_shrink_factors=(6, 4, 2, 1),
      aff_smoothing_sigmas=(3, 2, 1, 0)
    )
    self.transform(result['fwdtransforms'], fix)
    if jacobian: 
      jac = Data(ants.create_jacobian_determinant_image(fix, result['fwdtransforms'][0], do_log = True).numpy(), self)
      return jac
    if return_result: return result
