import argparse
import time as time
import calendar



class ParsingArguments:

    def parse_args(self):
        parser = argparse.ArgumentParser(description='the Predicting script allows users to choose the image_input, choose between cpu and gpu mode, checkpoint path and the topk up to 5')
        parser.add_argument('input', action='store')
        parser.add_argument('checkpoint', action='store')
        parser.add_argument('--top_k', dest='top_k', type=int, default=1)
        parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
        parser.add_argument('--gpu', action='store_true', default=False)
        return parser.parse_args()


    def parse_training_args(self):
        ''' Defining the user-friendly command-line interface.

            [1] https://docs.python.org/2/library/argparse.html
            [2]
        ''' 
        print('\n Author: Mohammad Salim Ghadbnan\n Udacity AI Nanodegree 2018\n', time.asctime()+'\n\n')
        parser = argparse.ArgumentParser(description='The training script allows users to choose the architecture, and the data directory. It also allows users to set hyperparameters for learning rate, number of hidden units, training epochs, batch size, cpu or gpu mode and finally savin the checkpoint')
        parser.add_argument('data_dir', action='store', help='Setting the images directory')
        parser.add_argument('--checkpoint', dest="checkpoint")
        parser.add_argument('--arch', dest="arch", default="vgg16", type=str, choices=["vgg16", "resnet34",'alexnet', 'densenet121', 'inception_v3'])
        parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.001)
        parser.add_argument('--hidden_units', dest="hidden_units", type=int, default=512)
        parser.add_argument('--epochs', dest="epochs", type=int, default=10)
        parser.add_argument('--gpu', action="store_true", default=False)
        parser.add_argument('--batch_size', dest="batch_size", default=64)
    
        return parser.parse_args()