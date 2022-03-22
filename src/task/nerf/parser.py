
import argparse


@classmethod
def set_parser(cls,parser:argparse.ArgumentParser):
    pass

    # parser.add_argument('--experiment-name','-e',
    #                     default="DEFAULT_EXPERIMENT",
    #                     type=str,
    #                     help="Name of experiment(Recommended to 'always' use).")

    # parser.add_argument('--save','-s',
    #                     action='store_true',
    #                     help="Use this flag to save the model after train. Mostly likely meaningless if you are not using the script to train a model")

    # parser.add_argument('--load','-l',
    #                     type=str,
    #                     help='PATH at which your model is saved.')

    # parser.add_argument('--train')