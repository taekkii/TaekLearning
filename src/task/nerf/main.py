
from . import parser
from .. import Task

from .train import NeRFTrainer
from . import video
from .utils import get_intrinsic

class NerfTask(Task):
    
    set_parser = parser.set_parser
    
    def __init__(self,args):
        
        super().__init__(args)
        
        print('=========== [NeRFTask] ==========')



    def __call__(self):
    
            
        if self.settings['train']:
            t = NeRFTrainer(self.datasets,self.models,self.trainchunks,self.settings)
            t.run()
        if self.settings['test']:
            net_c = self.models.get('nerf',self.models.get('nerf_c',None))
            net_f = self.models.get('nerf_f',None)
            
            #temporary
            
            for data in self.datasets.values():
                sample_img,_,fov = data[0]
                break
            # _,h,w = sample_img.shape
            # K = get_intrinsic(h,w,fov).to(sample_img.device)
            h,w=800,800
            K=get_intrinsic(h,w,fov).to(sample_img.device)
            video.spherical_record(self.settings['experiment_name'],
                                   net_c,net_f,800,800,K,frames=40,
                                   n_samples_coarse= self.settings['n_samples'],
                                   n_samples_fine  = self.settings['n_samples_fine'],
                                   t_near          = self.settings['t_near'],
                                   t_far           = self.settings['t_far'])