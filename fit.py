from tqdm import tqdm as tqdm
import torch
import os
from utils import AverageMeter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score,precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
import json



class Trainer(object):
    def __init__(self, args, loader, model, criterion, optimizer, logger, out_dir,writter, gpu, scheduler, custom):
        self.max_epochs = args.epochs
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.out_dir = out_dir
        self.writer_dict= writter
        self.gpu = gpu
        self.n_classes = args.n_classes
        self.scheduler = scheduler
        self.custom = custom
        self.class_names = ['elefante', 'farfalla', 'mucca', 'pecora', 'scoiattolo']
        self.device = 'cuda:'+str(0) if torch.cuda.is_available() else "cpu"


    def load_checkpoint(self, ckp=None, mode='test'):
        if mode == 'train':
            self.logger.info("Loading checkpoint for training..")
            checkpoint = torch.load(ckp)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_dict'])
            return checkpoint['epoch']
        else:
            self.logger.info("Loading checkpoint for testing..")
            checkpoint = torch.load(ckp)
            self.model.load_state_dict(checkpoint['state_dict'])
            return 
        

    def calculate_metrics(self, name, y_true, outputs, y_pred, visual_dir):
        print(len(y_true))
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(self.class_names)),
            yticks=np.arange(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i,                 
                    f"{cm[i, j]:d}",      
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        plt.tight_layout()
        cm_path = os.path.join(visual_dir,"confusion_matrix_final")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        

        acc = accuracy_score(y_true, y_pred)

        pre_m = precision_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
        rec_m = recall_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
        f1_m = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
        clasRep = classification_report(y_true=y_true, y_pred=y_pred, target_names=self.class_names,output_dict=False, zero_division=0)
        y_true_ohe = label_binarize(y_true, classes=list(range(len(self.class_names))))
        aucs = roc_auc_score(y_true_ohe, outputs, average=None)
        msg = (
                f"Classifier: {name} finished with:\n"
                f"  • Accuracy (overall):   {acc*100:.2f}%\n"
                f"  • Precision (macro):    {pre_m*100:.2f}%\n"
                f"  • Recall (macro):       {rec_m*100:.2f}%\n"
                f"  • F1 score (macro):     {f1_m*100:.2f}%"
            )
        

        plt.figure(figsize=(8, 6))
        for i, class_name in enumerate(self.class_names):
            
            fpr, tpr, _ = roc_curve(y_true_ohe[:, i], outputs[:, i])
            plt.plot(
                fpr, tpr,
                label=f"{class_name} (AUC = {aucs[i]:.2f})",
                linewidth=2
            )

        
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-All ROC Curves")
        plt.legend(loc="lower right", fontsize="small")
        plt.grid(True)
        plt.tight_layout()

        
        os.makedirs(visual_dir, exist_ok=True)
        plt.savefig(os.path.join(visual_dir, "roc_curves.png"), bbox_inches="tight")
        plt.show()
        self.logger.info(msg)
        self.logger.info("\n" + clasRep)
        y_true_ohe = label_binarize(y_true, classes=list(range(len(self.class_names))))
   
        return [acc, pre_m, rec_m, f1_m, clasRep]
    

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch + 1,
            'state_dict' : self.model.state_dict(),
            'optim_dict' : self.optimizer.state_dict(),
            }
        torch.save(state, os.path.join(self.out_dir + '/' + "best_checkpoint.tar"))

    def training_step(self):

        losses = AverageMeter()
        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['train_epoch']
        self.model.train()
        for img, lbl in tqdm(self.loader['Train_dl']):
            if self.gpu:
                img, lbl = img.to(self.device), lbl.to(self.device)

            if self.custom:
                feats, output = self.model(img)
            else: 
                output = self.model(img)

            loss = self.criterion(output, lbl)
            # print(loss)
            # print(lbl)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if self.scheduler:
            #     self.scheduler.step()
            losses.update(loss.item(), lbl.size(0))

        writer.add_scalar('train_loss', losses.avg, global_steps)
        self.writer_dict['train_epoch'] = global_steps + 1
        return losses
    
    def evaluation_step(self):

        losses = AverageMeter()
        correct = 0.0
        total = 0.0
        self.model.eval()
        with torch.no_grad():
            for img, lbl in tqdm(self.loader['Valid_dl']):
                if self.gpu:
                    img, lbl = img.to(self.device), lbl.to(self.device)

                if self.custom:
                    feats, output = self.model(img)
                else: 
                    output = self.model(img)
                preds = output.data.max(1)[1]

                loss = self.criterion(output, lbl)

                correct += (lbl.data == preds).sum()
                total += lbl.size(0)
                losses.update(loss.item(), lbl.size(0))
            
            acc = correct * 100 / total

        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['valid_epoch']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_Acc', acc, global_steps)
        self.writer_dict['valid_epoch'] = global_steps + 1
        return losses, acc
    
    def test_step(self, ckp, visual_dir):
        self.logger.info("Testing Step...")
        y_pred = np.empty((0, self.n_classes), float)
        y_true = []
        feats_all = []
        correct = 0.0
        total = 0.0
        self.load_checkpoint(ckp=ckp, mode='test')
        self.model.eval()
        with torch.no_grad():
            for img, lbl in tqdm(self.loader['Test_dl_labeled']):
                if self.gpu:
                    img, lbl = img.to(self.device), lbl.to(self.device)

                if self.custom:
                    feats, output = self.model(img)
                else: 
                    output = self.model(img)
                preds = output.data.max(1)[1]

                # print(feats.flatten(start_dim=1) .shape)
                feats_all.append(feats.flatten(start_dim=1) )
                y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                y_true.append(lbl.item())
                correct += (lbl.data == preds).sum()
                total += lbl.size(0)
                
            predic = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])
            metrics = self.calculate_metrics(name='VGG16', y_true=y_true, outputs=y_pred, y_pred=predic, visual_dir=visual_dir)
            acc = correct * 100 / total

       
        return metrics, torch.stack(feats_all), predic
    

    def train_loop(self, resume=False):
        if resume:
            init_epoch = self.load_checkpoint()
        else:
            init_epoch = 0

        best_acc = 0.0
        loss_tr_, loss_val_, acc_val_ = [], [], []
        for epoch in range(init_epoch, self.max_epochs):

            self.logger.info(f"Training Epochs {epoch + 1}/{self.max_epochs}...")

            loss_tr = self.training_step()

            loss_vl, acc_vl = self.evaluation_step()

            self.logger.info("Training Loss: {:.6f} ({:.6f}) and Validation Loss: {:.6f} ({:.6f}) / Validation Acc: {}".format(
                loss_tr.val, loss_tr.avg, loss_vl.val, loss_vl.avg, acc_vl
            ))
            loss_tr_.append(loss_tr.avg), loss_val_.append(loss_vl.avg), acc_val_.append(acc_vl)
            if best_acc < acc_vl:
                self.save_checkpoint(epoch)
                self.logger.info(f"Best model is saved at epoch {epoch + 1}")
                best_acc = acc_vl

            # if self.scheduler:
            #     self.scheduler.step(loss_vl.avg)
        
        return [loss_tr_, loss_val_, acc_val_]
