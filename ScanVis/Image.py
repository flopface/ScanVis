from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from ipywidgets import interact, fixed
from skimage.transform import rotate
from typing import Literal
from .useful_stuff import *
from matplotlib.colors import LinearSegmentedColormap, to_rgb

class Image():
  def __init__(self, id, data = [], keys = [], dont_normalise = False, cmap = 'inferno'):
    self.arrays = dict()

    # Data validation
    if type(data) is np.ndarray:
      if len(data.shape) == 3: data = [data]
      elif len(data.shape) != 4: raise TypeError(f'Input data has the wrong dimensions - shape = {data.shape}')
    elif type(data) != list: data = list(data)

    #Key validation
    if type(keys) not in [list, np.ndarray]: keys = [keys]
    if not all([type(key) is str for key in keys]): raise TypeError(f'Keys must all be strings')
    if len(keys) != len(data): print('Warning : number of input data is not equal to number of input keys, some may be excluded')

    for datum, key in zip(data, keys): self.read_data_source(datum, key)
    self.normalise_colors(dont_normalise)

    self.id = id
    self.cmap = cmap

  def read_data_source(self, datum, key):
    if type(datum) is np.ndarray: 
      if len(datum.shape) != 3: TypeError(f'Input data array for {key} has the wrong dimensions - shape = {datum.shape}')
      self.arrays[key] = datum
    elif type(datum) is str:
      if datum[-4:] == '.nii': self.arrays[key] = sitk.GetArrayFromImage(sitk.ReadImage(datum))
      elif datum[-4:] == '.npy': self.arrays[key] = np.load(datum)
    else: raise TypeError(f'Input data for {key} must be path to .nii or .npy, or an array, not {datum}')

  def normalise_color(self, key, centre = None):
    smallest, biggest = np.min(self.arrays[key]), np.max(self.arrays[key])
    if centre != None:
      if biggest > -smallest: smallest = (2*centre)-biggest
      else: biggest = (2*centre)-smallest
    self.normalisation_dict[key] = [smallest, biggest]

  def normalise_colors(self, dont_normalise):
    self.normalisation_dict = dict()
    if dont_normalise:
      self.normalisation_dict = dict()
      for key in self.arrays: self.normalisation_dict[key] = [None, None]
      return
    for key in self.arrays: self.normalise_color(key)
  
  def check_ax(self, ax, nx, ny, figsize, dpi, pad):
    if ax is None:
      ax_exists = False
      fig, ax = plt.subplots(ncols = nx, nrows = ny, figsize = figsize, dpi = dpi)
      fig.patch.set_facecolor('black')
      fig.tight_layout(pad = pad)
    else:
      ax_exists = True
      fig = None
    return ax, ax_exists, fig
  
  def plot(self, image = 0, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', seg_image = 'seg', slice = 128, structure_id = [], title = None, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, ax = None, figsize = (5,  5), dpi = 100, save = None, plot_legend = True, return_all = False):
    
    if type(structure_id) not in [list, np.ndarray]: structure_id = [structure_id]
    if type(c) not in [list, np.ndarray]: c = [c]
    if type(image) is int: image = list(self.arrays.keys())[image]
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi, 0)

    if view == 'Saggittal': 
      picture = rotate(self.arrays[image][:,:,255-slice], 270, preserve_range=True, order = 0)
      if len(structure_id) != 0 or return_all: seg = rotate(self.arrays[seg_image][:,:,255-slice], 270, preserve_range=True, order = 0)
    elif view == 'Axial': 
      picture = np.flip(self.arrays[image][:,slice,:])
      if len(structure_id) != 0 or return_all: seg = np.flip(self.arrays[seg_image][:,slice,:])
    else: 
      picture = np.flip(self.arrays[image][slice,:,:], 1)
      if len(structure_id) != 0 or return_all: seg = np.flip(self.arrays[seg_image][slice,:,:], 1)

    ax.imshow(picture, aspect = 1, cmap = self.cmap, vmin = self.normalisation_dict[image][0], vmax = self.normalisation_dict[image][1])
    
    for s, col in zip(structure_id, c):
      col = list(to_rgb(col))
      fill_cmap = LinearSegmentedColormap.from_list('my_cmap', [[0,0,0,0], col+[fill_alpha]], 2)
      fill, x, y = find_structure_and_outline(seg, s)
      ax.plot(x, y, 's', c = col, ms = ms, label = lut[s] if s in lut else None)
      ax.imshow(fill, cmap = fill_cmap)

    if plot_legend and (len(structure_id) > 0): ax.legend(labelcolor = 'white', facecolor = 'k', markerscale = 2, loc = 'upper right')
    ax.set(yticks = [], xticks = [])
    ax.set_xlabel(f'Slice {slice} - {self.id} - {image.capitalize()} - {view}', c = 'w' if title is None else title, fontsize = 8)
    if return_all: return ax, picture, seg
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive(self, image = 0, seg_image = 'seg', ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, figsize = (5, 5), dpi = 100):
    if seg_image not in self.arrays: 
      print(f'Warning : Segmentation \'{seg_image}\' does not exist')
      c = []
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def plot_str(view, {"".join([f"structure_{i+1} = {2+i}, " for i in range(N)])}slice = 100):\n'\
      f'  plot(image, view, seg_image, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], None, ms, c, fill_alpha, outline_alpha, None, figsize, dpi, None, True)\n\n'\
      f'interact(plot_str, view = [\'Saggittal\', \'Axial\', \'Coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    exec(func_str, {'plot' : self.plot, 'interact' : interact, 'image' : image, 'seg_image' : seg_image, 'ms' : ms, 'c' : c, 'fill_alpha' : fill_alpha, 'outline_alpha' : outline_alpha, 'figsize' : figsize, 'dpi' : dpi})

  def overlay(self, image = 0, view : Literal['Saggittal', 'Axial', 'Coronal'] = 'Saggittal', seg_image = 'seg', slice = 128, structure_id = [], title = None, ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, ax = None, figsize = (10, 5), dpi = 100, save = None, plot_legend = True):
    if structure_id is None: structure_id = []
    if type(structure_id) not in [list, np.ndarray]: structure_id = [structure_id]
    if type(c) not in [list, np.ndarray]: c = [c]*len(structure_id)
    ax, ax_exists, _ = self.check_ax(ax, 2, 1, figsize, dpi, 0)
    ax[1], _, seg = self.plot(image, view, seg_image, slice, structure_id, title, ms, c, fill_alpha, outline_alpha, ax[1], None, None, None, plot_legend, True)
    structures, counts = np.unique(seg, return_counts=True)

    inds = np.argsort(structures)
    ax[0].barh([lut[struct] + ' (' + str(struct) + ')' for struct in structures[inds][1:]], counts[inds][1:], height=0.9, align='center', color='r')
    ax[0].set_title('Relative volumes', c = 'w')
    ax[0].set_facecolor('black')
    ax[0].tick_params(axis='x', colors='white')
    ax[0].tick_params(axis='y', colors='white')
    ax[0].spines['top'].set_color('white')
    ax[0].spines['right'].set_color('white')
    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['left'].set_color('white')
    ax[0].set_xticks([])
    ax[0].yaxis.set_tick_params(labelcolor='white')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive_overlay(self, image = 0, seg_image = 'seg', ms = 2, c = 'w', fill_alpha = 0.2, outline_alpha = 1, figsize = (10, 5), dpi = 100):
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def overlay_str(view, {"".join([f"structure_{i+1} = {i+2}, " for i in range(N)])}slice = 100):\n'\
      f'  overlay(image, view, seg_image, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], None, ms, c, fill_alpha, outline_alpha, None, figsize, dpi, None, True)\n\n'\
      f'interact(overlay_str, view = [\'Saggittal\', \'Axial\', \'Coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'overlay' : self.overlay, 'interact' : interact, 'image' : image, 'seg_image' : seg_image, 'ms' : ms, 'c' : c, 'fill_alpha' : fill_alpha, 'outline_alpha' : outline_alpha, 'figsize' : figsize, 'dpi' : dpi})
