from .Image import *
from collections.abc import MutableMapping

class Images(MutableMapping):
  def __init__(self, images, keys, id = 'A0001', seg = None):
    if type(images) not in [list, np.ndarray]: images = [images]
    if type(keys) not in [list, np.ndarray]: keys = [keys]
    for i in range(len(images)): 
      if type(images[i]) is str: images[i] = Image(images[i])
      elif type(images[i]) is not Image: raise TypeError('All input images must be of type Image')
    if len(keys) != len(images): raise Exception('Different number of images and keys')

    self.images = dict()

    for image, key in zip(images, keys): 
      self.images[key] = image
      image.key = key
      image.id = id

    if seg is not None: self.set_seg(seg)

  def set_seg(self, seg):
    for image in self.images.values(): 
      image.set_seg(seg)
      if image.mask: image.mask_image()

  def __getitem__(self, key) -> Image:
    return self.images[key]
  
  def __setitem__(self, key, image : Image):
    self.images[key] = image
    
  def __iter__(self):
    return iter(self.images)

  def keys(self):
    return self.images.keys()

  def values(self):
    return self.images.values()
  
  def items(self):
    return self.images.items()

  def __delitem__(self, key):
    del self.images[key]
  
  def __len__(self):
    return len(self.images)
  
  def __dir__(self):
    # Include dictionary keys in autocompletion
    return super().__dir__() + list(self.images.keys())
  
  def __getattr__(self, name) -> Image:
    if name in self.images:
        return self.images[name]
    raise AttributeError(f"'Images' object has no attribute '{name}'")