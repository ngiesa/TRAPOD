from math import gamma
import pytorch_lightning as pl
import torch.optim as optim
from model.losses import sigmoid_focal_loss, weighted_binary_cross_entropy, add_l1_penalty
from torchmetrics import AUROC, AveragePrecision
from ray import tune
import torch
import numpy as np
from tqdm.auto import tqdm
from process.data_module import DataSet
from random import choices
import torchmetrics
from sklearn import metrics
import scipy.stats as st

class ModelTrainer(pl.LightningModule):
    def __init__(self, model, prevalence=0.1, criterion = sigmoid_focal_loss, gamma=1, 
                    tunable=False, lr=1e-4, l1_lambda=0.001, with_static=False):
        super().__init__()
        self.model = model
        self.prevalence = prevalence
        self.criterion = criterion
        self.tunable = tunable
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = l1_lambda
        self.with_static = with_static
        if "11." in torchmetrics.__version__:
            self.auroc = AUROC(task="binary") # task=binary
            self.auprc = AveragePrecision(task="binary")# pos_label=1
        else:
            self.auroc = AUROC()
            self.auprc = AveragePrecision(pos_label=1)

    def add_l1_norm_penalty(self, loss, model):
        if ".mlp_model.MLP" in str(type(self.model)):
            l1_norm = sum(abs(p).sum() for p in model.layers[0].parameters())
        elif "Combined" in str(type(self.model)):
            l1_norm = sum(abs(p).sum() for p in model.mlp.layers[0].parameters())
        else:
            l1_norm = 0
        return loss # + self.weight_decay * l1_norm TODO

    def forward(self, x, labels=None, static = None, model = None):
        if ".mlp_model.MLP" in str(type(self.model)): # TODO CHECK 
            out = model.propagate(x.squeeze(1)) 
        elif self.with_static:
            out = model(x, static)
        else:
            out = model(x)
        loss = 0
        if labels is not None:
            labels = labels.unsqueeze(1).float()
            if self.criterion == sigmoid_focal_loss:
                loss = self.criterion(
                    out, labels, self.prevalence, self.gamma
                )
            else:
                loss = self.criterion(
                    out, labels, self.prevalence
                )
            loss = self.add_l1_norm_penalty(loss=loss, model=model)
        return loss, out

    def training_step(self, batch, batch_idx):
        static = None
        sequences = batch["sequence"]
        labels = batch["label"]
        if self.with_static:
            static = batch["static"]
            #print(static.size()) # torch.Size([64, 1, 339])
        loss, out = self.forward(sequences, labels, static, model=self.model)
        out = out.squeeze(1)
        auroc = self.auroc(out, labels) 
        auprc = self.auprc(out, labels) 
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_auroc", auroc, prog_bar=True, logger=True)
        self.log("train_aupr", auprc, prog_bar=True, logger=True)
        return {"loss": loss, "auroc": auroc, "auprc": auprc}

    def validation_step(self, batch, batch_idx):
        static = None    
        sequences = batch["sequence"]
        labels = batch["label"]
        if self.with_static:
            static = batch["static"]
        one_count = list(np.array(labels.cpu().numpy())).count(1)
        # handle no positive samples in batch
        loss, out = self.forward(sequences, labels, static, model=self.model)
        out = out.squeeze(1)
        auroc = self.auroc(out, labels)
        # report auprc if no pos samples in batch
        if one_count == 0:
            auprc = torch.tensor(0.0)
        else:
            auprc = self.auprc(out, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_auroc", auroc, prog_bar=True, logger=True)
        self.log("val_aupr", auprc, prog_bar=True, logger=True)
        if self.tunable:
            tune.report(val_loss=loss.cpu().numpy(),
                        val_auroc=auroc.cpu().numpy(), 
                        val_auprc=auprc.cpu().numpy())
        return {"loss": loss, "auroc": auroc, "auprc": auprc}

    def testing_step(self, batch, batch_idx):
        static = None
        sequences = batch["sequence"]
        labels = batch["label"]
        if self.with_static:
            static = batch["static"]
        loss, out = self.forward(sequences, labels, static, model=self.model)
        out = out.squeeze(1)
        auroc = self.auroc(out, labels)
        auprc = self.auprc(out, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_auroc", auroc, prog_bar=True, logger=True)
        self.log("test_auprc", auprc, prog_bar=True, logger=True)
        if self.tunable:
            tune.report(test_loss=loss.cpu().numpy(), 
                        test_auroc=auroc.cpu().numpy(), 
                        test_auprc=auprc.cpu().numpy())
        return {"loss": loss, "auroc": auroc, "auprc": auprc, "out": out, "labels":labels}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def test_data(self, test_dataset, train_dataset, n_bootstrapps=1): 
        static = None
        prefix, res = {0:"train", 1:"val"}, {}
        for i, dataset in enumerate([train_dataset, test_dataset]):
            aurocs, auprcs = [], []
            
            n_bootstrapps = 100

            losses, n = [], n_bootstrapps
            for j in range(0, n):
                bootstrap = choices(dataset, k=len(dataset))
                set = DataSet(bootstrap)
                # perform bootstrapping n times 
                predictions, labels, loss_list = [], [], []
                precs, recs, fprs, tprs = [], [], [], []
                thres_aurocs, thres_auprcs = [], []
                for item in tqdm(set):
                    sequence = item["sequence"].unsqueeze(dim=0)
                    if self.with_static:
                        static = item["static"].unsqueeze(dim=0)
                    label = torch.tensor(item["label"])
                    loss, out = self.forward(x=sequence, labels = None, static=static, model=self.model)
                    loss = self.criterion(out, torch.unsqueeze(torch.unsqueeze(label, 0), 0), self.prevalence, self.gamma)
                    predictions.append(out.item())
                    labels.append(label.item())
                    loss_list.append(loss.item())
                predictions_torch = torch.tensor(predictions)
                labels_torch = torch.tensor(labels)
                aurocs.append(self.auroc(predictions_torch, labels_torch).item()); auprcs.append(self.auprc(predictions_torch, labels_torch).item())
                losses.append(np.mean(loss_list))
                fpr, tpr, thres_auroc = metrics.roc_curve(np.array(labels), np.array(predictions))
                prec, rec, thres_auprc = metrics.precision_recall_curve(np.array(labels), np.array(predictions))
                fprs.append(list(fpr)); tprs.append(list(tpr))
                precs.append(list(prec)); recs.append(list(rec))
                thres_aurocs.append(list(thres_auroc)); thres_auprcs.append(list(thres_auprc))

            
            auprcs_CI = st.t.interval(0.95, len(auprcs)-1, loc=np.mean(auprcs), scale=st.sem(auprcs))
            aurocs_CI = st.t.interval(0.95, len(aurocs)-1, loc=np.mean(aurocs), scale=st.sem(aurocs))
            loss_CI = st.t.interval(0.95, len(losses)-1, loc=np.mean(losses), scale=st.sem(losses))
            
            res["{}_aurocs".format(prefix[i])] = aurocs; res["{}_auprcs".format(prefix[i])] = auprcs; 
            
            res["{}_auroc_CI_lower".format(prefix[i])] = aurocs_CI[0]; res["{}_auroc_CI_upper".format(prefix[i])] = aurocs_CI[1]
            res["{}_auprc_CI_lower".format(prefix[i])] = auprcs_CI[0]; res["{}_auprc_CI_upper".format(prefix[i])] = auprcs_CI[1]
            res["{}_mean_auroc".format(prefix[i])] = np.mean(aurocs); res["{}_mean_auprc".format(prefix[i])] = np.mean(auprcs)
            res["{}_fprs".format(prefix[i])] = fprs[0]; res["{}_tprs".format(prefix[i])]  = tprs[0]
            res["{}_precs".format(prefix[i])]  = precs[0]; res["{}_recs".format(prefix[i])]  = recs[0]
            res["{}_thres_aurocs".format(prefix[i])] = thres_aurocs[0]; res["{}_thres_auprcs".format(prefix[i])] = thres_auprcs[0]
            res["{}_mean_loss".format(prefix[i])] = np.mean(losses)
            res["{}_loss_CI_lower".format(prefix[i])] = loss_CI[0]; res["{}_loss_CI_upper".format(prefix[i])] = loss_CI[1]


        return res