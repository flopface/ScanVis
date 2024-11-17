import numpy as np

pop = '/Users/work/Desktop/MPhys/popty-ping/'
mphys = '/Users/work/Desktop/MPhys/'
labbook = '/Users/work/Desktop/MPhys/images_for_lab_book/'

with open('/Users/work/Desktop/Code/MPhys/LUT.txt', 'r') as file: lines = file.readlines()
lut = dict()
rlut = dict()
for line in lines:
  if line[0].isdigit():
    line = [word for word in line.split() if word != ' ']
    lut[int(line[0])] = line[1]
    rlut[line[1]] = int(line[0])

info_data = '/Users/work/Desktop/MPhys/freesvol01.txt'

with open(info_data, 'r') as file: lines = file.readlines()

patient_dict = dict()
for line in lines[2:]:
  line = line.split('\t')
  patient_dict[line[4][1:-1]] = [int(line[7][1:-1]), line[8][1:-1]]

def find_edges_1D(arr):
  string_arr = ''.join(arr.astype(int).astype(str))
  edge_arr = string_arr.replace('01', '02').replace('10', '20') # Turn any True by a False to a 2
  edge_arr = (np.array(list(edge_arr)).astype(int) - 2).astype(bool)
  return ~edge_arr

def find_edges_2D(arr):
  x_arr = np.array([find_edges_1D(line) for line in arr])
  y_arr = np.swapaxes(np.array([find_edges_1D(line) for line in np.swapaxes(arr, 0, 1)]),0,1)
  edge_arr = x_arr | y_arr
  return edge_arr

def find_structure_edges(image, structure_id):
  image = ~(image - structure_id).astype(bool)
  image = find_edges_2D(image)
  return image

def find_structure_coords(image, structure_id):
  image = find_structure_edges(image, structure_id)
  y, x = np.where(image)
  return x, y

def progress_word(current, total, word = 'lets just say, hehe, my peenitz'):
  length = len(word)
  progress = int(current / total * length // 1)
  print('\r|' + word[:progress] + ' '*int(length-progress), end = f'| {100*current/total:.2f}%')