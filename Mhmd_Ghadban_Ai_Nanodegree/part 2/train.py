from ParseArguments import ParsingArguments
from deviceSystem import deviceSystem
from  DataInformation import data_dir, dataset_sizes
from ModelTraining import AlexNet, VGGNet, DenseNet, ResNet, Inception
import time as time
from torchvision import models



def main():
	Parseargs = ParsingArguments()
	args = Parseargs.parse_training_args()
	dv = deviceSystem()
	device = dv.gpu_device(args.gpu)
	model = getattr(models, args.arch)(pretrained=True)
	
	print(' This training are running in {} mode\n'.format(device))
	print(' Folder: {}\n Dataset Sizez: {}\n Architecture used: {}\n Epochs {}\n Learning_rate {}\n Batch size {}\n Ckekpoint path {}\n'
		  .format(args.data_dir, dataset_sizes, args.arch, args.epochs, args.learning_rate, args.batch_size, args.checkpoint))
	
	if args.arch == 'alexnet':
		dns = AlexNet()
		dns.run(device, args, model)

	elif args.arch == 'vgg16':
		vgg = VGGNet()
		vgg.run(device, args, model)

	elif args.arch == 'densenet121':
		dns121 = DenseNet()
		dns121.run(device, args, model)

	elif args.arch == 'resnet34':
			rsnt = ResNet()
			rsnt.run(device, args, model)

	elif args.arch == 'inception_v3':
			incpt = Inception()
			incpt.run(device, args, model)
		
if __name__ == '__main__':
	main()
