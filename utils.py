import logging
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def create_logger(args, model_name, phase='train'):
    root_output_dir = Path(args.out_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = args.dataset_name
    model = model_name
    

    final_output_dir = root_output_dir / dataset / model_name

    print('--> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(args.log_dir) / dataset / model / \
            (model_name + '_' + time_str)
    print('--> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Visualizer:
    def __init__(self, save_dir):
        self.visual_dir = save_dir
        

    def plot_losses(self, losses, filename="loss_curve1.png"):
   
        os.makedirs(self.visual_dir, exist_ok=True)

        epochs = range(1, len(losses[0]) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs,losses[0], label="Train Loss")
        plt.plot(epochs, losses[1], label="Valid Loss")
        plt.title("Training vs. Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        
        plt.show()

       
        save_path = os.path.join(self.visual_dir, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved loss curve to {save_path}")

    
    def plot_PCA(self,feats,y_true, filename="PCA_plot.png"):
        feats = feats.squeeze(1)
        X = feats.detach().cpu().numpy()     
        y = np.array(y_true)        

        class_names = ['elefante', 'farfalla', 'mucca', 'pecora', 'scoiattolo']
       

       
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)  

      
        plt.figure(figsize=(8, 8))
        for cls in np.unique(y):
            idxs = (y == cls)
            plt.scatter(
                X_pca[idxs, 0],
                X_pca[idxs, 1],
                label=class_names[cls],
                alpha=0.6
            )

        plt.legend()
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of VGG16 Feature Representations")
        plt.grid(True)
        plt.tight_layout()

       
        os.makedirs(self.visual_dir, exist_ok=True)
        path = os.path.join(self.visual_dir, "feature_pca.png")
        plt.savefig(path, bbox_inches="tight")
        plt.show()



def get_mean_std(loader=None):
    mean, std = 0.0, 0.0
    for img, lbl in loader:

        batch_samples = img.size(0) 
        img = img.view(batch_samples, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean,std
