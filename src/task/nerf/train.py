
import trainer
from . import ray
from . import render


def squared_error(c_gt,c_render):
    return ((c_gt-c_render)**2).sum()
class NeRFTrainer(trainer.Trainer):

    #override
    def start(self):
        pass

    #override
    def step(self,data):
        train_chunk = self.trainchunks[0] # temporary

        gt_imgs,poses = data
        gt_imgs = gt_imgs.to(self.settings['device'])
        poses   = poses.to(self.settings['device'])
        
        
        rendered_imgs, mse_loss = train_chunk.forward_and_backward(poses , target=gt_imgs , obj_func=squared_error)
        
        

        
    
    def validate(self):
        pass
        