import torch
from torch.nn import BCELoss
from plants_sm.ml.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import torchmetrics
import lightning as L

from ._multi_modal_model import NPClassifierDNN

class InteractionModelTransferLearning(nn.Module):

    def __init__(self, transfer_learning=True) -> None:
        super(InteractionModelTransferLearning, self).__init__()
        self.compounds_model = NPClassifierDNN()
        self.proteins_model = DNN(1280, [2560, 5120], 5743, batch_norm=True, last_sigmoid=True, dropout=None)
        self.interaction_model = DNN(1024, [1024, 1024, 512, 256], 1, batch_norm=True, last_sigmoid=True, dropout=None)
        self.compounds_model_fp = DNN(6144, [3072, 1536], 1536, batch_norm=True, last_sigmoid=False, dropout=None)

        self.transfer_learning = transfer_learning
        if self.transfer_learning:
            self.compounds_head = nn.Sequential(
                nn.Linear(1536, 768),
                nn.ReLU(),
                nn.BatchNorm1d(768),
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.BatchNorm1d(384),
            )

            self.proteins_head = nn.Sequential(
                nn.Linear(5120, 2560),
                nn.ReLU(),
                nn.BatchNorm1d(2560),
                nn.Linear(2560, 1280),
                nn.ReLU(),
                nn.BatchNorm1d(1280),
                nn.Linear(1280, 640),
                nn.ReLU(),
                nn.BatchNorm1d(640)
            )

        else:
            self.compounds_head = nn.Sequential(
                nn.Linear(6144, 3072),
                nn.ReLU(),
                nn.Linear(3072, 1536),
                nn.ReLU(),
            )

            self.proteins_head = nn.Sequential(
                nn.Linear(1280, 640),
                nn.ReLU(),
                nn.Linear(640, 320),
                nn.ReLU(),
            )
            self.interaction_model = DNN(1856, [1024, 1024, 512, 256], 1, batch_norm=True, last_sigmoid=True, dropout=None)



    def forward(self, x_compounds, x_proteins, return_predictions=False):
        if self.transfer_learning:
            with torch.no_grad():
                np_prediction, compounds_embedding = self.compounds_model([x_compounds], return_embedding=True)
                protein_ec_number_prediction, protein_embedding = self.proteins_model([x_proteins], return_embedding=True)
            
            compounds_embedding = self.compounds_head(compounds_embedding)
            protein_embedding = self.proteins_head(protein_embedding)
        else:
            compounds_embedding = self.compounds_head(x_compounds)
            protein_embedding = self.proteins_head(x_proteins)

        # compound_process = self.compounds_model_fp([x_compounds])
        interaction = torch.cat([compounds_embedding, protein_embedding], dim=1)
        interaction = self.interaction_model([interaction])

        if return_predictions:
            return interaction, np_prediction, protein_ec_number_prediction
        else:
            return interaction
    
class InteractionModel(L.LightningModule):

    def __init__(self, scheduler=None, protein_model_path="", compounds_model_path="", transfer_learning=True) -> None:
        self.scheduler = scheduler
        self.protein_model_path = protein_model_path
        self.compounds_model_path = compounds_model_path
        self.transfer_learning = transfer_learning

        super().__init__()
        self._create_model()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.problem_type = "binary"

    def _update_constructor_parameters(self):
        self._contructor_parameters.update({
            'metric': self.metric,
        })

    def load_models(self):
        np_classifier_model_pretrained_dict = torch.load(self.compounds_model_path)
        np_classifier_model_pretrained_dict = {k.replace("np_classifier_model.", ""): v 
                                               for k, v in np_classifier_model_pretrained_dict["state_dict"].items() 
                                               if k.replace("np_classifier_model.", "") 
                                               in self.interaction_model.compounds_model.state_dict()}
        
        self.interaction_model.compounds_model.load_state_dict(np_classifier_model_pretrained_dict)
        self.interaction_model.proteins_model.load_state_dict(torch.load(self.protein_model_path))

    def _create_model(self):
        self.interaction_model = InteractionModelTransferLearning(transfer_learning=self.transfer_learning)
        self.load_models()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=0.001)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x_compounds, x_proteins):
        return self.interaction_model(x_compounds, x_proteins)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)
    
    def training_step(self, batch, batch_idx):
        x_proteins, x_compounds, y = batch
        logits = self(x_compounds, x_proteins)
        loss = self.compute_loss(logits, y)
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        
        accuracy = self.accuracy(logits, y)
        f1 = self.f1(logits, y)
        recall = self.recall(logits, y)
        precision = self.precision(logits, y)

        self.log("train_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("train_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        self.log("train_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_proteins, x_compounds, target = batch
        output = self(x_compounds, x_proteins)

        output = output.detach().cpu()
        target = target.detach().cpu()

        loss = self.compute_loss(output, target)
        accuracy = self.accuracy(output, target)
        f1 = self.f1(output, target)
        recall = self.recall(output, target)
        precision = self.precision(output, target)

        self.log("val_loss", loss.item(), on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", accuracy, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        return loss
        

    def predict_step(self, batch):
        if len(batch) == 3:
            x_proteins, x_compounds, target = batch
        else:
            x_proteins, x_compounds = batch
        return self(x_compounds, x_proteins)