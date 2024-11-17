import SimpleITK as sitk
import os
import numpy as np

from .useful_stuff import *

class Subject:
  def __init__(self, seg_file, folder = '/Users/work/Desktop/MPhys/popty-ping/segmentations'):
    options = [file for file in os.listdir(folder) if seg_file in file]
    if len(options) == 0: raise Exception(f'No patient with that ID')
    elif len(options) > 1: print('Available scans:', options)
    
    self.seg_file = options[0]

    seg = sitk.ReadImage(os.path.join(folder, self.seg_file))
    self.spacing = seg.GetSpacing()
    #self.pixel_vol = self.spacing[0]*self.spacing[1]*self.spacing[2]/1000
    self.pixel_vol = seg.GetSizeOfPixelComponent()/1000
    seg = sitk.GetArrayFromImage(seg)
    self.total_volume = np.sum(seg.astype(bool).astype(int))*self.pixel_vol
    
    structures, counts = np.unique(seg, return_counts = True)
    self.id = self.seg_file[:5]
    self.vols = dict()
    if self.id in patient_dict: self.age, self.gender = patient_dict[self.id]
    else: self.age, self.gender = 0, 'U'
    counts = counts*self.pixel_vol
    for s, c in zip(structures, counts): self.vols[s] = c

  def __getitem__(self, key):
    return self.vols[key]
  
  def __str__(self):
    return f'{self.id}'
  
  def print(self, verbose = False):
    print(f'{self.id} | {"Girl" if self.gender == "F" else " Boy"} | {' ' if self.age < 120 else ''}{self.age // 12} years {' ' if self.age % 12 < 10 else ''}{self.age % 12} months | {' ' if self.total_volume < 1000 else ''}{self.total_volume:.1f}cm\u00b3 | {self.seg_file}')
