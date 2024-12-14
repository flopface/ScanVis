from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from ipywidgets import interact, fixed
from skimage.transform import rotate

from .useful_stuff import *

class Scan:
  def __init__(self, scan_file, seg_file, age = 0, gender = 'Unknown', cmap = 'inferno'):
    self.scan_file = scan_file
    self.seg_file = seg_file
    if not os.path.isfile(self.scan_file): raise Exception(f'Scan \'{self.scan_file}\' does not exist')
    if not os.path.isfile(self.seg_file): raise Exception(f'Segmentation \'{self.seg_file}\' does not exist')
    scan = sitk.ReadImage(self.scan_file)
    seg = sitk.ReadImage(self.seg_file)
    try: self.get_arrays(scan, seg)
    except: self.allign(scan, seg)
    self.id = os.path.split(self.scan_file)[1][:-4]
    self.age, self.gender = age, gender
    self.spacing = scan.GetSpacing()
    self.direction = np.array(scan.GetDirection()).reshape(3, 3)
    self.origin = scan.GetOrigin()
    self.voxel_volume = self.spacing[0]*self.spacing[1]*self.spacing[2]
    self.volume = np.sum(self.arrays['mask']) * self.voxel_volume
    self.cmap = cmap

  def set_array(self, key, array):
    self.arrays[key] = array

  def allign(self, scan, seg):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(scan)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    seg = resample.Execute(seg)
    self.get_arrays(scan, seg)

  def get_arrays(self, scan, seg):
    self.arrays = {'scan' : sitk.GetArrayFromImage(scan), 'seg' : sitk.GetArrayFromImage(seg)}
    self.arrays['shifted_seg'] = self.arrays['seg'] + 35*(self.arrays['seg'].astype(bool).astype(int))
    self.arrays['mask'] = self.arrays['seg'].astype(bool).astype(int)
    self.arrays['brain'] = self.arrays['scan'] * self.arrays['mask']
    self.normalise_color()

  def normalise_color(self):
    self.normalisation_dict = dict()
    for key in self.arrays: self.normalisation_dict[key] = [np.min(self.arrays[key]), np.max(self.arrays[key])]

  def imvi(self, image, view):
    if type(image) == int: image = ['scan', 'brain', 'shifted_seg', 'mask'][image]
    else: image = image.lower()
    if type(view) == int: view = ['saggittal', 'axial', 'coronal'][view]
    else: view = view.lower()
    return image, view
  
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
  
  def plot(self, image = 'scan', view = 'saggittal', slice = 128, structure_id = None, title = None, ms = 2, c = 'w', ax = None, figsize = (5, 5), dpi = 100, save = None, print_plot = False, plot_legend = True):
    if structure_id is None: structure_id = []
    if type(structure_id) not in [list, np.ndarray]: structure_id = [structure_id]
    if type(c) not in [list, np.ndarray]: c = [c]*len(structure_id)
    image, view = self.imvi(image, view)
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi, 0)

    if view == 'saggittal': picture, seg = rotate(self.arrays[image][:,:,255-slice], 270, preserve_range=True), rotate(self.arrays['seg'][:,:,255-slice], 270, preserve_range=True)
    elif view == 'axial': picture, seg = np.flip(self.arrays[image][:,slice,:]), np.flip(self.arrays['seg'][:,slice,:])
    else: picture, seg = np.flip(self.arrays[image][slice,:,:], 1), np.flip(self.arrays['seg'][slice,:,:], 1)
    ax.imshow(picture, aspect = 1, cmap = self.cmap, vmin = self.normalisation_dict[image][0], vmax = self.normalisation_dict[image][1])
    for s, c in zip(structure_id, c): ax.plot(*(find_structure_coords(seg, s)), 's', c = c, ms = ms, label = lut[s] if s in lut else None)
    '''    for s in structure_id:
      truth = ~(seg - s).astype(bool)
      stuff = picture[truth]
      if len(stuff): print(np.mean(stuff))'''
    if title is None: title = f'Slice {slice} - {self.id} - {image.capitalize()} - {view}'
    if plot_legend and (len(structure_id) > 0): ax.legend(labelcolor = 'white', facecolor = 'k', markerscale = 2, loc = 'upper right')
    ax.set(yticks = [], xticks = [])
    ax.set_xlabel(title, c = 'w', fontsize = 8)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive(self, image = 'scan', ms = 2, c = 'w', figsize = (5, 5), dpi = 100):
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def plot_str(view, {"".join([f"structure_{i+1}, " for i in range(N)])}slice):\n'\
      f'  plot(image, view, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], None, ms, c, None, figsize, dpi, None, True)\n\n'\
      f'interact(plot_str, view = [\'saggittal\', \'axial\', \'coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'plot' : self.plot, 'interact' : interact, 'image' : image, 'ms' : ms, 'c' : c, 'figsize' : figsize, 'dpi' : dpi})

  def overlay(self, image = 'scan', view = 'saggittal', slice = 128, structure_id = None, title = None, ms = 2, c = 'w', ax = None, figsize = (10, 5), dpi = 100, save = None, print_plot = False):
    if structure_id is None: structure_id = []
    if type(structure_id) not in [list, np.ndarray]: structure_id = [structure_id]
    if type(c) not in [list, np.ndarray]: c = [c]*len(structure_id)
    image, view = self.imvi(image, view)
    ax, ax_exists, _ = self.check_ax(ax, 2, 1, figsize, dpi, 0)
    ax[1] = self.plot(image, view, slice, structure_id, title, ms, c, ax[1])

    if view == 'saggittal': structures, counts = np.unique(self.arrays['seg'][:,:,slice].flatten(), return_counts=True)
    elif view == 'axial': structures, counts = np.unique(self.arrays['seg'][:,slice,:].flatten(), return_counts=True)
    else: structures, counts = np.unique(self.arrays['seg'][slice,:,:].flatten(), return_counts=True)
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
    plt.show()
    if print_plot: print(f'.plot(\'{image}\',\'{view}\',slice={slice},structure_id={structure_id},figsize={figsize[0]/2,figsize[1]},dpi={dpi},ms={ms},c=\'{c}\')')

  def interactive_overlay(self, image = 'scan', ms = 2, c = 'w', figsize = (10, 5), dpi = 100):
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def overlay_str(view, {"".join([f"structure_{i+1}, " for i in range(N)])}slice):\n'\
      f'  overlay(image, view, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], None, ms, c, None, figsize, dpi, None, True)\n\n'\
      f'interact(overlay_str, view = [\'saggittal\', \'axial\', \'coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'overlay' : self.overlay, 'interact' : interact, 'image' : image, 'ms' : ms, 'c' : c, 'figsize' : figsize, 'dpi' : dpi})

  def plot_three(self, image = 'scan', slices = [128, 128, 128], structure_id = None, title = None, ms = 2, c = 'w', ax = None, figsize = (12, 5), dpi = 100, pad = -2, save = None, plot_legend = True):
    ax, ax_exists, _ = self.check_ax(ax, 3, 1, figsize, dpi, pad)
    for i in [0,1,2]: ax[i] = self.plot(image, i, slices[i], structure_id, None, ms, c, ax[i], figsize, dpi, save, False, plot_legend = i is 2 and plot_legend)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def plot_hella_slices(self, image = 'scan', view = 'saggittal', slices = 5, structure_id = None, ms = 2, c = 'w', ax = None, figsize = (2,5), dpi = 100, pad = -2, save = None, label_slices = True, label_images = False, plot_legend = True):    
    image, view = self.imvi(image, view)
    if type(slices) == int:
      slicer = 'np.any(self.arrays[\'mask\'][:, :, i])' if view == 'saggittal' else 'np.any(self.arrays[\'mask\'][:,i,:])' if view == 'axial' else 'np.any(self.arrays[\'mask\'][i,:,:])'
      i = 0
      while not eval(slicer): i += 1
      start = i
      while eval(slicer): i += 1
      slices = np.linspace(start+10, i-10, slices).astype(int)
    ax, ax_exists, _ = self.check_ax(ax, len(slices), 1, [figsize[0]*len(slices), figsize[1]], dpi, pad)
    for i, slice in enumerate(slices): ax[i] = self.plot(image, view, slice, structure_id, f'Slice {slice}' if label_slices else None if label_images else '', ms, c, ax[i], plot_legend = plot_legend and (i == len(slices)-1))
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def plot_buttloads_of_slices(self, image = 'scan', slices = 5, structure_id = None, ms = 2, c = 'w', ax = None, figsize = (3,3), dpi = 100, pad = -2, save = None, label_images = False, plot_legend = True, title = None):    
    ax, ax_exists, fig = self.check_ax(ax, slices, 3, [figsize[0]*slices, figsize[1]*3], dpi, pad)
    for i in [0,1,2]: ax[i] = self.plot_hella_slices(image, i, slices, structure_id, ms, c, ax[i], figsize, dpi, pad, None, label_slices = (i == 2 and not label_images), label_images=label_images, plot_legend = (i == 0) and plot_legend)
    ax[0][0].set_ylabel('saggittal', color = 'w')
    ax[1][0].set_ylabel('axial', color = 'w')
    ax[2][0].set_ylabel('coronal', color = 'w')
    if title != None and fig != None: fig.suptitle(title, c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def compare_buttloads_of_slices(self, other : Scan, image = 'scan', slices = 5, structure_id = None, ms = 2, c = 'w', ax = None, figsize = (3,3), dpi = 100, pad = -2, save = None, label_slices = True, plot_legend = False, title = None):    
    ax, ax_exists, fig = self.check_ax(ax, slices, 6, [figsize[0]*slices, figsize[1]*6], dpi, pad)
    for i in [0,2,4]: ax[i] = self.plot_hella_slices(image, int(i/2), slices, structure_id, ms, c, ax[i], figsize, dpi, pad, None, label_slices, plot_legend = plot_legend and i == 0)
    for i in [1,3,5]: ax[i] = other.plot_hella_slices(image, int((i-1)/2), slices, structure_id, ms, c, ax[i], figsize, dpi, pad, None, label_slices, plot_legend = False)
    ax[0][0].set_ylabel(str(self.id)+' saggittal', color = 'w')
    ax[1][0].set_ylabel(str(other.id)+' saggittal', color = 'w')
    ax[2][0].set_ylabel(str(self.id)+' axial', color = 'w')
    ax[3][0].set_ylabel(str(other.id)+' axial', color = 'w')
    ax[4][0].set_ylabel(str(self.id)+' coronal', color = 'w')
    ax[5][0].set_ylabel(str(other.id)+' coronal', color = 'w')
    if title != None and fig != None: fig.suptitle(title, c = 'w')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()
    
  def print(self):
    print(f'Patient ID : {self.id}')
    print(f'       Age : {self.age // 12} years {self.age % 12} months')
    print(f'    Gender : {self.gender}')
    print(f' Scan File : {self.scan_file}')
    print(f'  Seg File : {self.seg_file}')
    print(f' Scan size : {self.scan_array.shape}')
    print(f'   Spacing : {self.spacing}')
    print(f'    Volume : {self.volume/1000:.1f}cm\u00b3')
  
  def compare(self, other : Scan, image = 'scan', view = 'saggittal', slice = 128, structure_id = None, title = None, ms = 2, c = 'w', ax = None, figsize = (5, 5), dpi = 100, pad = -1, save = None, print_info = False, plot_legend = True):
    if print_info:
      self.print()
      print()
      other.print()
    ax, ax_exists, _ = self.check_ax(ax, 2, 1, figsize, dpi, pad)
    self.plot(image, view, slice, structure_id, title, ms, c, ax[0], plot_legend=plot_legend)
    other.plot(image, view, slice, structure_id, title, ms, c, ax[1], plot_legend=plot_legend)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive_compare(self, other : Scan, image = 'scan', ms = 2, c = 'w', figsize = (10, 5), dpi = 100, pad = -1):
    N = len(c) if type(c) != str else 1
    func_str = \
      f'def comp_str(view, {"".join([f"structure_{i+1}, " for i in range(N)])}slice):\n'\
      f'  comp(other, image, view, slice, [{"".join([f"structure_{i+1}, " for i in range(N)])[:-2]}], None, ms, c, None, figsize, dpi, pad, None, False)\n'\
      f'interact(comp_str, view = [\'saggittal\', \'axial\', \'coronal\'], {"".join([f"structure_{i+1} = (0,77,1), " for i in range(N)])}slice = (0,255,1))'
    
    exec(func_str, {'comp' : self.compare, 'interact' : interact, 'other' : other, 'image' : image, 'ms' : ms, 'c' : c, 'figsize' : figsize, 'dpi' : dpi, 'pad' : pad})

  def compare_rgb(self, other : Scan, image = 'brain', view = 'saggittal', slice = 128, structure_id = None, ax = None, figsize = (5, 5), dpi = 100, save = None, print_info = False, plot_legend = True):
    if print_info:
      self.print()
      print()
      other.print()
    image, view = self.imvi(image, view)
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi, 0)

    rgb = np.zeros((256, 256, 3))

    if view == 'saggittal':
      rgb[:,:,0] = rotate(self.arrays[image][:,:,slice], 270, preserve_range = True)
      rgb[:,:,1] = rotate(other.arrays[image][:,:,slice], 270, preserve_range = True)
      rgb[:,:,2] = rotate(self.arrays[image][:,:,slice], 270, preserve_range = True)
      for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
      ax.imshow(rgb)
      if structure_id != None: 
        ax.plot(*find_structure_coords(rotate(self.arrays['seg'][:,:,slice], 270, preserve_range = True), structure_id), 's', c = [1,0,1,1], ms = 2, label = f'P{self.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
        ax.plot(*find_structure_coords(rotate(other.arrays['seg'][:,:,slice], 270, preserve_range = True), structure_id), 's', c = [0,1,0,1], ms = 2, label = f'P{other.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
    elif view == 'axial': 
      rgb[:,:,0] = np.flip(self.arrays[image][:,slice,:])
      rgb[:,:,1] = np.flip(other.arrays[image][:,slice,:])
      rgb[:,:,2] = np.flip(self.arrays[image][:,slice,:])
      for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
      ax.imshow(rgb)
      if structure_id != None: 
        ax.plot(*find_structure_coords(np.flip(self.arrays['seg'][:,slice,:]), structure_id), 's', c = [1,0,1,1], ms = 2, label = f'P{self.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
        ax.plot(*find_structure_coords(np.flip(other.arrays['seg'][:,slice,:]), structure_id), 's', c = [0,1,0,1], ms = 2, label = f'P{other.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
    else: 
      rgb[:,:,0] = self.arrays[image][slice,:,:]
      rgb[:,:,1] = other.arrays[image][slice,:,:]
      rgb[:,:,2] = self.arrays[image][slice,:,:]
      for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
      ax.imshow(rgb)
      if structure_id != None: 
        ax.plot(*find_structure_coords(self.arrays['seg'][slice,:,:], structure_id), 's', c = [1,0,1,1], ms = 2, label = f'P{self.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
        ax.plot(*find_structure_coords(other.arrays['seg'][slice,:,:], structure_id), 's', c = [0,1,0,1], ms = 2, label = f'P{other.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
    if plot_legend: ax.legend()
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def compare_three_rgb(self, other : Scan, image = 'brain', slices = [128, 128, 128], structure_id = None, title = None, ms = 2, c = 'w', ax = None, figsize = (12, 5), dpi = 100, pad = -2, save = None, plot_legend = True):
    ax, ax_exists, _ = self.check_ax(ax, 3, 1, figsize, dpi, pad)
    for i in [0,1,2]: ax[i] = self.compare_rgb(other, image, i, slices[i], structure_id, ax[i], figsize, dpi, save, False, i == 2 and plot_legend)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def interactive_compare_rgb(self, other : Scan, image = 'brain', ms = 2, structure_id = False, figsize = (10, 5), dpi = 100):
    if structure_id: interact(self.compare_rgb, other = fixed(other), image = fixed(image), view = ['saggittal', 'coronal', 'axial'], slice = (0, 255, 1), structure_id = (0, 78, 1), ax = fixed(None), figsize = fixed(figsize), dpi = fixed(dpi), save = fixed(None), print_info = fixed(False))
    else: interact(self.compare_rgb, other = fixed(other), image = fixed(image), view = ['saggittal', 'coronal', 'axial'], slice = (0, 255, 1), structure_id = fixed(None), ax = fixed(None), figsize = fixed(figsize), dpi = fixed(dpi), save = fixed(None), print_info = fixed(False))
