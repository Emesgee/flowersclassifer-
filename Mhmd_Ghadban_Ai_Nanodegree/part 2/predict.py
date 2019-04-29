from CheckPointManupulationSystem import CheckPointManupulationSystem
from  ParseArguments import ParsingArguments
from ImageProcessingAndClassification import imagePredictionSystem
from deviceSystem import deviceSystem
from time import time


def main():
    PrseArgs=ParsingArguments()
    args = PrseArgs.parse_args()
    dv = deviceSystem()
    device = dv.gpu_device(args.gpu)
    ckpoint=CheckPointManupulationSystem()
    model = ckpoint.load_checkpoint(args.checkpoint)
    imgPredSys=imagePredictionSystem()
    model = model.to(device)
    image = args.input
    cat_to_name = imgPredSys.load_categories(args.category_names)   
    probs, classes = imgPredSys.predict(image, model, device, args.top_k)        

    print(' This prediction are running in {} mode\n'.format(device))
    print(' Image input: {}\n Checkpoint Path : {}\n Top K: {}\n'
          .format(args.input, args.checkpoint, args.top_k))
    print(probs,[cat_to_name[name] for name in classes])
    
if __name__ == "__main__":
    main()
