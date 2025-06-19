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
from model_utils import MyVGG16


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

   
    model = MyVGG16(num_classes=5)

    for i, layer in enumerate(model.classif):
        if isinstance(layer, nn.Dropout):
            model.classif[i] = nn.Dropout(p=0)
            
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1,weight_decay=0)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1,weight_decay=0)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,        
    #                             patience=30, verbose=True)
    steps_per_epoch = len(loaders['Train_dl'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5,steps_per_epoch=steps_per_epoch,
    epochs=args.epochs,pct_start=0.3, div_factor=10, final_div_factor=100)
    mean, std = get_mean_std(loaders['Train_dl'])
    print(mean, std)


    visualizer = Visualizer(save_dir='./visual')
    trainer = Trainer(args, loaders, model, criterion, optimizer, logger, out_dir, 
                      writer_dict, True, scheduler, True)

    if args.isTrainable:
        metrics = trainer.train_loop()
        visualizer.plot_losses(losses=metrics, filename="normal_tr.png")
    else:

        metrics, feats, y_true = trainer.test_step(ckp = '/media/FastData/dimpap/test_wndb/Classification_Assingment/outputs/animals/VGG16/best_checkpointadam88.tar'
                                    ,visual_dir='./visual')
        
        visualizer.plot_PCA(feats=feats, y_true=y_true)
    writer = trainer.writer_dict['writer']
    writer.close()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animal_Dataset_T1", parents=[get_argparse()])
    args = parser.parse_args()
    main(args)