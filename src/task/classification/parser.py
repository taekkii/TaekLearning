

import argparse

@classmethod
def set_parser(cls,parser:argparse.ArgumentParser):
    parser.add_argument("--train",action='store_true',help='Use this script to train')
    parser.add_argument("--test",action='store_true',help="Use this script to test. Most likely meaningless when the model is neither trained or got its weight loaded")
    return parser