from .useful_stuff import *
import numpy as np
from typing import Literal
import SimpleITK as sitk
import os
from skimage.transform import rotate

class Array3D():
  def __init__(self, data):
    self.data = self.read_data_source(data)

  def read_data_source(self, data):
    self.isNone = False
    if type(data) is type(None): 
      data = np.zeros((256, 256, 256)).astype(int)
      self.isNone = True
    elif type(data) is np.ndarray:
      if len(data.shape) == 3: self.array = data
      else: raise TypeError(f'Input data must be 3D - current shape = {data.shape}')
    elif type(data) is str:
      if not os.path.isfile(data): raise Exception(f'Input data does not exist')
      if data[-4:] == '.nii': self.array = sitk.GetArrayFromImage(sitk.ReadImage(data))
      elif data[-4:] == '.npy': self.array = np.load(data)
    else: raise TypeError(f'Input data must be path to .nii or .npy, or an array, not {data}')

  def get_slice(self, view : Literal['Saggittal', 'Axial', 'Coronal'], slice):
    if view == 'Saggittal': picture = rotate(self.array[:,:,255-slice], 270, preserve_range=True, order = 0)
    elif view == 'Axial': picture = np.flip(self.array[:,slice,:])
    else: picture = np.flip(self.array[slice,:,:], 1)
    return picture
  
  def get_slices(self, view : Literal['Saggittal', 'Axial', 'Coronal'], slices, buffer):
    i = 0
    while not np.any(self.get_slice(view, i)): i += 1
    start = i
    i = 255
    while not np.any(self.get_slice(view, i)): i -= 1
    slices = np.linspace(start+buffer[0], i-buffer[1], slices).astype(int)
    return slices
