import os
from os.path import join
import numpy as np
from .Scan import *
from .useful_stuff import user_decision

class Database:
  def __init__(self, scan_folder, seg_folder):
    self.scan_folder = scan_folder
    self.seg_folder = seg_folder
    segs = [file for file in os.listdir(self.seg_folder) if '.nii' in file]
    self.files = [file for file in os.listdir(scan_folder) if '.nii' in file and file in segs]
    self.check_excess()
    
  def check_excess(self):
    scan_files = [file for file in os.listdir(self.scan_folder) if '.nii' in file]
    seg_files = [file for file in os.listdir(self.seg_folder) if '.nii' in file]
    self.extra_scans = [file for file in scan_files if file not in seg_files]
    self.extra_segs = [file for file in seg_files if file not in scan_files]
    if len(self.extra_scans) > 0: print('No segmentations found for:', *self.extra_scans)
    if len(self.extra_segs) > 0: print('No scans found for:', *self.extra_segs)

  def __call__(self, id):
    options = [file for file in self.files if id in file]
    if len(options) == 0: raise Exception(f'No patient with that ID')
    elif len(options) == 1: file = options[0]
    else: file = user_decision(options)
    return Scan(scan_file = join(self.scan_folder, file), seg_file = join(self.seg_folder, file), scan_folder = self.scan_folder, seg_folder = self.seg_folder)