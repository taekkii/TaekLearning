
from visdom import Visdom
import torch

class Visualizer:
        
        def __init__(self,env='main'):
            self.visdom = Visdom()
            self.env = env
            self.vis_dict = dict()
            
            self.visdom.close(env=env)

        def attach_graph(self, title:str , target , legend:str, **additional):
            if title not in self.vis_dict:
                self.new(title,'graph',**additional)

            vis_obj = self.vis_dict[title]
            vis_obj['opts']['legend'].append(legend)
            vis_obj['trace'].append(target)


        def new(self , title:str , category:str , **additional ):
            #----- [Delete previous object if id is occupied] -----#
            self.delete(title)
            if   category == 'graph': self.vis_dict[title] = self._config_graph(title,additional)
            elif category == 'text' : pass
            elif category == 'image': pass

            self.vis_dict[title].update(additional)

        def delete(self , title:str ):
            if title not in self.vis_dict: return
            if 'win' in self.vis_dict[title]:
                self.visdom.close( win=self.vis_dict[title]['win'] , env=self.env )
            self.vis_dict.pop(title)

        
        def run(self):
            for vis_obj in self.vis_dict.values():
                self._run_vis_obj(vis_obj)


        def _config_graph(self,title:str,additional:dict):
            win = self.visdom.line(Y=[0] , env=self.env)
            opts = dict( title=title , legend=[] )
            
            vis_obj = dict( category='graph' , win=win , opts=opts , trace=[] )
            vis_obj.update(additional)
            return vis_obj
        
        def _run_vis_obj(self,vis_obj:dict):
            if vis_obj['category'] == 'graph':
                win , opts =vis_obj['win'] , vis_obj['opts']
                trace = vis_obj['trace']
                
                if len(trace)==0:return

                x_len = len(trace[0])
                if x_len <= 0: return 
                
                x0    = vis_obj.get('x0',0) 
                x     = list(range(x0,x_len+x0))

                 
                if len(trace)==1:
                    self.visdom.line(Y=trace[0] , X=x , win=win , opts=opts , env=self.env )
                else:
                    self.visdom.line(Y=torch.Tensor(trace).t() , X=x , win=win , opts=opts , env=self.env )
