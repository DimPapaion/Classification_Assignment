from Dataset import CustomDataset
from config import get_argparse
import argparse
from fit import *
from tensorboardX import SummaryWriter
from utils import *
import pprint
import torchvision
import torch
import torch.nn as nn


def main(args):
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    
    CD = CustomDataset(args)

    loaders = CD.build_loader()
    
    print(loaders)
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    logger, out_dir, tensor_log_dir = create_logger(args, "VGG16")
    writer_dict = {
        'writer': SummaryWriter(tensor_log_dir),
        'train_epoch': 0,
        'valid_epoch': 0,
    }
    logger.info(pprint.pformat(args))

   
    model = torchvision.models.vgg16(weights=None)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=5)

    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Dropout):
            model.classifier[i] = nn.Dropout(p=0.0)
            
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1, weight_decay=0.0)
    mean, std = get_mean_std(loaders['Train_dl'])
    print(mean, std)
    visualizer = Visualizer(save_dir='./visual')
    trainer = Trainer(args, loaders, model, criterion, optimizer, logger, out_dir, writer_dict, True, None, None)

    metrics = trainer.train_loop()
   
    visualizer.plot_losses(losses=metrics, filename="adam_loss.png")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animal_Dataset_T1", parents=[get_argparse()])
    args = parser.parse_args()
    main(args)