from .Scant import *

class Scanter:
  def __init__(self, scan_folder, seg_folder):
    self.scan_folder = scan_folder
    self.seg_folder = seg_folder

  def __call__(self, scan_file, seg_file = None):
    return Scant(scan_file = scan_file, seg_file = seg_file, scan_folder = self.scan_folder, seg_folder = self.seg_folder)
