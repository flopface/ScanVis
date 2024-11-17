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
    self.shape = self.scan_array.shape
    self.volume = np.sum(self.mask_array.astype(int)) * self.voxel_volume
    self.cmap = cmap


  def allign(self, scan, seg):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(scan)  # Use the MRI scan as the reference
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for segmentation
    seg = resample.Execute(seg)  # Resample the segmentation
    self.get_arrays()

  def get_arrays(self, scan, seg):
    self.scan_array = sitk.GetArrayFromImage(scan)
    self.seg_array = sitk.GetArrayFromImage(seg)
    self.shifted_seg_array = self.seg_array + 15*(self.seg_array.astype(bool).astype(int))
    self.mask_array = self.seg_array.astype(bool)
    self.brain_array = self.scan_array * self.mask_array

  def imvi(self, image, view):
    if type(image) == int: image = ['scan', 'brain', 'shifted_seg', 'mask'][image]
    else: image = image.lower()
    if type(view) == int: view = ['Saggittal', 'Axial', 'Coronal'][view]
    else: view = view.capitalize()
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
  
  def plot(self, image = 'scan', view = 'saggittal', title = None, slice = 128, structure_id = None, ax = None, figsize = (5, 5), dpi = 100, ms = 2, save = None):
    
    image, view = self.imvi(image, view)
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi, 0)
    if view == 'Saggittal': 
      ax.imshow(rotate(eval(f'self.{image}_array[:,:,slice]'), 270, preserve_range=True), aspect = self.spacing[0]/self.spacing[1], cmap = self.cmap)
      if structure_id != None: ax.plot(*find_structure_coords(rotate(self.seg_array[:,:,slice], 270, preserve_range=True), structure_id), 'ws', ms = ms, label = lut[structure_id] if structure_id in lut else 'Unknown')
    elif view == 'Axial': 
      ax.imshow(np.flip(eval(f'self.{image}_array[:,slice,:]')), aspect = self.spacing[0]/self.spacing[2], cmap = self.cmap)
      if structure_id != None: ax.plot(*(find_structure_coords(np.flip(self.seg_array[:,slice,:]), structure_id)), 'ws', ms = ms, label = lut[structure_id] if structure_id in lut else 'Unknown')
    else: 
      ax.imshow(eval(f'self.{image}_array[slice,:,:]'), aspect = self.spacing[2]/self.spacing[1], cmap = self.cmap)
      if structure_id != None: ax.plot(*find_structure_coords(self.seg_array[slice,:,:], structure_id), 'ws', ms = ms, label = lut[structure_id] if structure_id in lut else 'Unknown')
    
    if title is None: title = f'{f'Structure {structure_id}: {lut[structure_id]}\n' if structure_id != None else ''}Slice {slice} - {self.id} - {image.capitalize()} - {view}'
    ax.set(yticks = [], xticks = [])
    ax.set_xlabel(title, c = 'w', fontsize = 8)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()
  
  def overlay(self, image = 'scan', view = 'saggittal', title = None, slice = 128, structure_id = 0, figsize = (8, 6), dpi = 100, ms = 2, color = 'w'):
    fig, ax = plt.subplots(ncols = 2, figsize = figsize, dpi = dpi, gridspec_kw={'width_ratios': [1, 1]})
    fig.patch.set_color('k')
    
    image, view = self.imvi(image, view)

    if title is None: title = f'P{self.id} - {image.capitalize()} - {view}'
    if view == 'Saggittal': 
      segs, counts = np.unique(self.seg_array[:,:,slice].flatten(), return_counts=True)
      ax[1].imshow(rotate(eval(f'self.{image}_array[:,:,slice]'), 270, preserve_range=True), aspect = self.spacing[0]/self.spacing[1], cmap = self.cmap)
      ax[1].plot(*find_structure_coords(rotate(self.seg_array[:,:,slice], 270, preserve_range=True), structure_id), 's', c = color, ms = ms, label = lut[structure_id] if structure_id in lut else 'Unknown')
    elif view == 'Axial': 
      segs, counts = np.unique(self.seg_array[:,slice,:].flatten(), return_counts=True)
      ax[1].imshow(np.flip(eval(f'self.{image}_array[:,slice,:]')), aspect = self.spacing[0]/self.spacing[2], cmap = self.cmap)
      ax[1].plot(*find_structure_coords(np.flip(self.seg_array[:,slice,:]), structure_id), 's', c = color, ms = ms, label = lut[structure_id] if structure_id in lut else 'Unknown')
    else: 
      segs, counts = np.unique(self.seg_array[slice,:,:].flatten(), return_counts=True)
      ax[1].imshow(eval(f'self.{image}_array[slice,:,:]'), aspect = self.spacing[2]/self.spacing[1], cmap = self.cmap)
      ax[1].plot(*find_structure_coords(self.seg_array[slice,:,:], structure_id), 's', c = color, ms = ms, label = lut[structure_id] if structure_id in lut else 'Unknown')
    
    inds = np.argsort(segs)

    ax[0].barh([lut[seg] + ' (' + str(seg) + ')' for seg in segs[inds][1:]], counts[inds][1:], height=0.9, align='center', color='r')
    
    ax[0].set_title('Relative volumes', c = 'w')
    ax[0].set_facecolor('black')  # Black background for bar chart
    ax[0].tick_params(axis='x', colors='white')  # White tick marks on x-axis
    ax[0].tick_params(axis='y', colors='white')  # White tick marks on y-axis
    ax[0].spines['top'].set_color('white')
    ax[0].spines['right'].set_color('white')
    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['left'].set_color('white')
    ax[0].set_xticks([])
    ax[0].yaxis.set_tick_params(labelcolor='white')  # White y-axis labels
    ax[1].axis('off')
    ax[1].set_title(title, c = 'w')
    if structure_id in lut: ax[1].legend(labelcolor = 'white', frameon = False, markerscale = 2)
    #ax[1] = self.plot(image, view, title, slice, ax[1])
    plt.show()

  def interactive_overlay(self, image = 'scan', title = None, figsize = (10, 6), dpi = 100, ms = 2, color = 'w'):
    interact(self.overlay, image = fixed(image), title = fixed(title), figsize = fixed(figsize), dpi = fixed(dpi),
             slice = (0, max(eval(f'self.{image}_array.shape'))-1, 1), structure_id = (0, 77, 1), view = ['Saggittal', 'Axial', 'Coronal'], ms = fixed(ms), color = fixed(color))
  
  def plot_three(self, slices = [128, 128, 128], image = 'scan', label_images = True, ax = None, figsize = (10, 4), dpi = 200, pad = -2, ms = 1, save = None, structure_id = None):
    ax, ax_exists, _ = self.check_ax(ax, 3, 1, figsize, dpi, pad)
    for i in [0,1,2]: ax[i] = self.plot(image, i, None if label_images else '', slices[i], structure_id, ax[i], ms = ms)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def plot_hella_slices(self, slices = 5, image = 'scan', view = 1, label_images = True, ax = None, figsize = (2, 5), dpi = 200, pad = -2, ms = 1, save = None, structure_id = None):
    
    image, view = self.imvi(image, view)

    if type(slices) == int:
      slicer = 'self.mask_array[:, :, i]' if view == 'Saggittal' else 'self.mask_array[:,i,:]' if view == 'Axial' else 'self.mask_array[i,:,:]'
      i = 0
      while not np.any(eval(slicer)): i += 1
      start = i
      while np.any(eval(slicer)): i += 1
      slices = np.linspace(start+10, i-10, slices).astype(int)
    
    ax, ax_exists, _ = self.check_ax(ax, len(slices), 1, [figsize[0]*len(slices), figsize[1]], dpi, pad)

    for i, slice in enumerate(slices): ax[i] = self.plot(image, view, f'Slice {slice}' if label_images else '', slice, structure_id, ax[i], ms=ms)
    
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches='tight')
    plt.show()

  def plot_buttloads_of_slices(self, n_slices = 5, image = 'scan', label_images = True, ax = None, figsize = (3, 3), dpi = 200, pad = 0, ms = 1, title = None, save = None, structure_id = None):
    ax, ax_exists, fig = self.check_ax(ax, n_slices, 3, [figsize[0]*n_slices, figsize[1]*3], dpi, pad)

    ax[0] = self.plot_hella_slices(slices = n_slices, image = image, view = 'Saggittal', label_images = label_images, ax = ax[0], structure_id = structure_id, ms = ms)
    ax[1] = self.plot_hella_slices(slices = n_slices, image = image, view = 'Axial', label_images = label_images, ax = ax[1], structure_id = structure_id, ms = ms)
    ax[2] = self.plot_hella_slices(slices = n_slices, image = image, view = 'Coronal', label_images = label_images, ax = ax[2], structure_id = structure_id, ms = ms)
    
    ax[0][0].set_ylabel('Saggittal', color = 'w')
    ax[1][0].set_ylabel('Axial', color = 'w')
    ax[2][0].set_ylabel('Coronal', color = 'w')

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
  
  def compare(self, other : Scan, image = 'scan', view = 'saggittal', title = None, slice = 128, structure_id = None, ms = 2, ax = None, figsize = (10, 5), dpi = 100, pad = 0, save = None, print_info = False):
    if print_info:
      self.print()
      print()
      other.print()
    ax, ax_exists, _ = self.check_ax(ax, 2, 1, figsize, dpi, pad)
    self.plot(image, view, title, slice, structure_id, ax[0], ms = ms)
    other.plot(image, view, title, slice, structure_id, ax[1], ms = ms)
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive_compare(self, other : Scan, image = 'scan', title = None, figsize = (10, 5), dpi = 100, pad = 0):
    interact(self.compare, other = fixed(other), image = fixed(image), view = ['Saggittal', 'Coronal', 'Axial'], title = fixed(title), slice = (0, 255, 1), structure_id = (0, 78, 1), ax = fixed(None), figsize = fixed(figsize), dpi = fixed(dpi), pad = fixed(pad), save = fixed(None), print_info = fixed(False))
    
  def compare_rgb(self, other : Scan, image = 'scan', view = 'saggittal', title = None, slice = 128, structure_id = None, ax = None, figsize = (5, 5), dpi = 100, save = None, print_info = False):
    if print_info:
      self.print()
      print()
      other.print()
    image, view = self.imvi(image, view)
    ax, ax_exists, _ = self.check_ax(ax, 1, 1, figsize, dpi, 0)

    if view == 'Saggittal':
      rgb = np.zeros(eval(f'list(rotate(self.{image}_array[:,:,slice], 270).shape) + [3]'))
      rgb[:,:,0] = rotate(eval( f'self.{image}_array[:,:,slice]'), 270, preserve_range = True)
      rgb[:,:,1] = rotate(eval(f'other.{image}_array[:,:,slice]'), 270, preserve_range = True)
      rgb[:,:,2] = rotate(eval( f'self.{image}_array[:,:,slice]'), 270, preserve_range = True)
      for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
      ax.imshow(rgb)
      if structure_id != None: 
        ax.plot(*find_structure_coords(rotate(self.seg_array[:,:,slice], 270, preserve_range = True), structure_id), 's', c = [1,0,1,1], ms = 2, label = f'P{self.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
        ax.plot(*find_structure_coords(rotate(other.seg_array[:,:,slice], 270, preserve_range = True), structure_id), 's', c = [0,1,0,1], ms = 2, label = f'P{other.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
    elif view == 'Axial': 
      rgb = np.zeros(eval(f'list(self.{image}_array[:,slice,:].shape) + [3]'))
      rgb[:,:,0] = np.flip(eval( f'self.{image}_array[:,slice,:]'))
      rgb[:,:,1] = np.flip(eval(f'other.{image}_array[:,slice,:]'))
      rgb[:,:,2] = np.flip(eval( f'self.{image}_array[:,slice,:]'))
      for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
      ax.imshow(rgb)
      if structure_id != None: 
        ax.plot(*find_structure_coords(np.flip(self.seg_array[:,slice,:]), structure_id), 's', c = [1,0,1,1], ms = 2, label = f'P{self.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
        ax.plot(*find_structure_coords(np.flip(other.seg_array[:,slice,:]), structure_id), 's', c = [0,1,0,1], ms = 2, label = f'P{other.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
    else: 
      rgb = np.zeros(eval(f'list(self.{image}_array[slice,:,:].shape) + [3]'))
      rgb[:,:,0] = eval( f'self.{image}_array[slice,:,:]')
      rgb[:,:,1] = eval(f'other.{image}_array[slice,:,:]')
      rgb[:,:,2] = eval( f'self.{image}_array[slice,:,:]')
      for i in [0,1,2]: rgb[:,:,i] /= np.max(rgb[:,:,i])*2
      ax.imshow(rgb)
      if structure_id != None: 
        ax.plot(*find_structure_coords(self.seg_array[slice,:,:], structure_id), 's', c = [1,0,1,1], ms = 2, label = f'P{self.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
        ax.plot(*find_structure_coords(other.seg_array[slice,:,:], structure_id), 's', c = [0,1,0,1], ms = 2, label = f'P{other.id} {lut[structure_id] if structure_id in lut else 'Unknown'}')
    if ax_exists: return ax
    if save != None: plt.savefig(save, dpi = 600, bbox_inches = 'tight')
    plt.show()

  def interactive_compare_rgb(self, other : Scan, image = 'scan', view = 'saggittal', title = None, figsize = (10, 5), dpi = 100, pad = 0):
    interact(self.compare_rgb, other = fixed(other), image = fixed(image), view = ['Saggittal', 'Coronal', 'Axial'], title = fixed(title), slice = (0, 255, 1), structure_id = (0, 78, 1), ax = fixed(None), figsize = fixed(figsize), dpi = fixed(dpi), save = fixed(None), print_info = fixed(False))