
import argparse


@classmethod
def set_parser(cls,parser:argparse.ArgumentParser):
    parser.add_argument("--train",action='store_true',help='Use this script to train')
    parser.add_argument("--test",action='store_true',help="Use this script to test. Most likely meaningless when the model is neither trained or got its weight loaded")
    
    parser.add_argument('--ray-batch-size',type=int, default=None, help='#of rays per iteration. 2^19 /  (n_samples+n_samples_fine) default')
    parser.add_argument("--t-near" , type=float , default=0. , help="hyperparameter t_near")
    parser.add_argument("--t-far" , type=float, default=None , 
                        help='hyperparameter t_far. [DEFAULT BEHAVIOR]: assumes object-centric learning, where center of the object is at (0,0,0).'+ 
                             'Default t_far will be automatically estimated to be 2 x euc_distance(the furthest camera origin), if not specified')
    
    parser.add_argument('--n-samples',type=int,default=64,help='Number of sample points/ray. In coarse-to-fine scenario, this is # of samples for coarse network')
    parser.add_argument('--n-samples-fine',type=int,default=0,help='Number of sample points/ray for a fine network. If >0, need 2 model configuration, '+
                                                                   'named nerf_c/nerf_f, + 2 trainchunks each')
    parser.add_argument('--coarse-ratio' , type=float,default=0.5,help='Coarse-fine ratio when integrating final rgb.'+
                                                                       ' i.e. rgb_final=[COARSE-RATIO]*rgb_coarse + (1-[COARSE-RATIO])*rgb_fine. Should range in 0~1')
    parser.add_argument('--center-crop-iteration','-cci', type=int,default=0,help='center crop iteration')
    parser.add_argument('--i-validation',type=int,default=500,help='Cycle of validation')
    
    return parser
