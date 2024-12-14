import os
from os.path import join
import numpy as np
from .Scan import *
from .Subject import *
from .useful_stuff import user_decision

class Database:
  def __init__(self, scan_folder, seg_folder, patient_info = None):
    self.scan_folder = scan_folder
    self.seg_folder = seg_folder
    segs = [file for file in os.listdir(self.seg_folder) if '.nii' in file]
    self.files = sorted([file for file in os.listdir(scan_folder) if '.nii' in file and file in segs])
    self.check_excess()
    self.validate_patient_info(patient_info)
    
  def check_excess(self):
    scan_files = [file for file in os.listdir(self.scan_folder) if '.nii' in file]
    seg_files = [file for file in os.listdir(self.seg_folder) if '.nii' in file]
    self.extra_scans = [file for file in scan_files if file not in seg_files]
    self.extra_segs = [file for file in seg_files if file not in scan_files]
    if len(self.extra_scans) > 0: print('No segmentations found for:', *self.extra_scans)
    if len(self.extra_segs) > 0: print('No scans found for:', *self.extra_segs)

  def validate_patient_info(self, input_patient_info):
    self.patient_info = dict()
    if input_patient_info is None:
      for file in self.files: self.patient_info[file] = [0, 'Unknown']
      return
    female = ['f', 'w', 'g', 'female', 'woman', 'girl']
    male = ['m', 'b', 'male', 'man', 'boy']
    for key, val in input_patient_info.items():
      if type(val) not in [list, np.ndarray]:
        print(f'Warning: item {key} in info dictionary is not a list')
        input_patient_info[key] = [0, 'Unknown']
      elif (len(val) != 2) or (type(val[0]) not in [int, float]) or ((val[1].lower() not in male) and (val[1].lower() not in female)):
        print(f'Warning: item {key} in info dictionary must be formatted [\'age\', \'gender\']')
        input_patient_info[key] = [0, 'Unknown']

    for file in self.files:
      matching = [key for key in input_patient_info if key in file]
      if len(matching) > 1: 
        print(f'Warning: Multiple matches in info dictionary for {file}')
        self.patient_info[file] = input_patient_info[user_decision(matching)]
      elif len(matching) == 1: self.patient_info[file] = input_patient_info[matching[0]]
      else: 
        print(f'Warning: No match in info dictionary for {file}')
        self.patient_info[file] = [0, 'Unknown']

  def scan(self, id):
    options = [file for file in self.files if id in file]
    if len(options) == 0: raise Exception(f'No patient with that ID')
    elif len(options) == 1: file = options[0]
    else: file = user_decision(options)
    return Scan(scan_file = join(self.scan_folder, file), seg_file = join(self.seg_folder, file), age = self.patient_info[file][0], gender = self.patient_info[file][1])
  
  def subject(self, id):
    options = [file for file in self.files if id in file]
    if len(options) == 0: raise Exception(f'No patient with that ID')
    elif len(options) == 1: file = options[0]
    else: file = user_decision(options)
    return Subject(seg_file = join(self.seg_folder, file), age = self.patient_info[file][0], gender = self.patient_info[file][1])

  def get_subjects(self):
    subjects = []
    for i, file in enumerate(self.files):
      progress_word(i+1, len(self.files))
      subjects.append(self.subject(file))
    return subjects

  def __call__(self, id):
    return self.scan(id)