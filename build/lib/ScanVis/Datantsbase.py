from .Database import *
from .Scant import *
from .Data import *
from os import listdir
from os.path import join, split

class Datantsbase(Database):
  def __init__(self, scan_folder, seg_folder, jacobian_folder, transform_folder, patient_info = None):
    super().__init__(scan_folder, seg_folder, patient_info)
    self.jacobian_folder = jacobian_folder
    self.transform_folder = transform_folder

  def scant(self, id):
    options = [file for file in self.files if id in file]
    if len(options) == 0: raise Exception(f'No patient with that ID')
    elif len(options) == 1: file = options[0]
    else: file = user_decision(options)
    return Scant(scan_file = join(self.scan_folder, file), seg_file = join(self.seg_folder, file), age = self.patient_info[file][0], gender = self.patient_info[file][1])
  
  def get_jacobian(self, id, jacobian_file = None):
    if jacobian_file == None:
      pid = id[:id.find('_')] if '_' in id else id
      options = [file for file in os.listdir(self.jacobian_folder) if pid in file]
      if len(options) == 0: raise Exception(f'No patient with that ID')
      elif len(options) == 1: jacobian_file = options[0]
      else: jacobian_file = user_decision(options)
      jacobian = np.load(join(self.jacobian_folder, jacobian_file))
    else: jacobian = np.load(jacobian_file)
    jacobian = np.flip(np.swapaxes(jacobian, 0, 2), 1)
    jac = Data(jacobian, self.scan(id))
    return jac
  
  def form_jacobians(self, warp_to):
    fix = self.scant(warp_to)
    fix.array2ants(fix.arrays['brain'])
    for file in listdir(self.transform_folder)[0]:
      transforms = listdir(join(self.transform_folder, file))
      warp = transforms[0] if 'Warp' in transforms[0] else transforms[1]
      jac = ants.create_jacobian_determinant_image(fix, warp)
      print(jac)

  def warp_jacobians(self, warp_to, save_folder):
    fix = self.scant(warp_to)
    fix = fix.array2ants(fix.arrays['brain'])
    progress_word(0, len(listdir(self.transform_folder)))      
    for i, file in enumerate([file for file in listdir(self.transform_folder) if '.' not in file]):
      if file+'.npy' not in listdir(save_folder):
        mov = self.scant(file).array2ants(np.load(join(self.jacobian_folder, file+'.npy')).astype(float)*1000)
        transforms = [join(self.transform_folder, file, t) for t in listdir(join(self.transform_folder, file))]
        jac = ants.apply_transforms(fixed=fix, moving=mov, transformlist=transforms,  interpolator='linear')
        np.save(join(save_folder, file), jac.numpy()/1000)
      progress_word(i+1, len(listdir(self.transform_folder)))      

  def run_registrations(self, to_warp, warp_to):
    fix = self.scant(warp_to)
    for i, scan in enumerate(to_warp):
      if scan not in listdir(self.transform_folder):
        mov = self.scant(scan)
        result = mov.register(fix)
        transforms = result['fwdtransforms']
        os.makedirs(join(self.transform_folder, scan), exist_ok=True)
        for t in transforms: os.rename(t, join(self.transform_folder, scan, split(t)[1]))
      progress_word(i+1, len(to_warp), 'The International Criminal Court issues arrest warrants for Israeli prime minister Benjamin Netanyahu (pictured), former Israeli defense minister Yoav Gallant, and Hamas leader Mohammed Deif in its investigation of war crimes in Palestine.')      
  
  def __call__(self, id):
    return self.scant(id)
  

  