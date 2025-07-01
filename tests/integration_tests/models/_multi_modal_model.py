import torch
from torch.nn import BCELoss
from plants_sm.ml.models.lightning_model import InternalLightningModule
from plants_sm.ml.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
import lightning as L


class NPClassifierDNN(nn.Module):
    def __init__(self, num_categories=730, original=False):
        super(NPClassifierDNN, self).__init__()
        self.original = original
        if not self.original:
            self.dense1 = nn.Linear(6144, 2048)
            self.bn1 = nn.BatchNorm1d(2048)
            self.dense2 = nn.Linear(2048, 3072)
            self.bn2 = nn.BatchNorm1d(3072)
            self.dense3 = nn.Linear(3072, 1536)
            self.bn3 = nn.BatchNorm1d(1536)
            self.dense4 = nn.Linear(1536, 1536)
            self.dropout = nn.Dropout(0.2)
            self.output = nn.Linear(1536, num_categories)
        
        else:
            self.dense1 = nn.Linear(6144, 3072)
            self.bn1 = nn.BatchNorm1d(3072)
            self.dense2 = nn.Linear(3072, 2304)
            self.bn2 = nn.BatchNorm1d(2304)
            self.dense3 = nn.Linear(2304, 1152)
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(1152, num_categories)

    def forward(self, x, return_embedding=False):
        if not self.original:
            x = F.relu(self.dense1(x[0]))
            x = self.bn1(x)
            x = F.relu(self.dense2(x))
            x = self.bn2(x)
            x = F.relu(self.dense3(x))
            x = self.bn3(x)
            x = self.dense4(x)
            x = F.relu(x)  # This is the output of the fourth dense layer
            embedding = torch.clone(x)
            x = self.dropout(embedding)
            x = self.output(x)
        else:
            x = F.relu(self.dense1(x[0]))
            x = self.bn1(x)
            x = F.relu(self.dense2(x))
            x = self.bn2(x)
            x = F.relu(self.dense3(x))
            embedding = torch.clone(x)
            x = self.dropout(x)
            x = self.output(x)
        
        x = torch.sigmoid(x)
        if return_embedding:
            return x, embedding
        else:
            return x

class ReactionsECModel(nn.Module):
    
        def __init__(self) -> None:
            super(ReactionsECModel, self).__init__()
            self.reactions_ec_model = nn.Sequential(
                nn.Linear(1200, 308),
                nn.LogSoftmax(dim=1)
            )

            self.embedding_part = nn.Sequential(
                nn.Linear(2400, 1200),
                nn.BatchNorm1d(1200),
                nn.ReLU()
            )

        def forward(self, x, return_embedding=False):
            x = self.embedding_part(x)
            if return_embedding:
                embedding = torch.clone(x)
                return self.reactions_ec_model(x), embedding
            else:
                return self.reactions_ec_model(x)
            

class MultiModalTransferLearning(nn.Module):

    def __init__(self, transfer_learning=True) -> None:
        super(MultiModalTransferLearning, self).__init__()
        self.compounds_model = NPClassifierDNN()
        self.proteins_model = DNN(1280, [2560, 5120], 5743, batch_norm=True, last_sigmoid=True, dropout=None)
        self.compounds_model_fp = DNN(6144, [3072, 1536], 1536, batch_norm=True, last_sigmoid=False, dropout=None)
        self.batch_normalization_hidden = nn.BatchNorm1d(1200)
        self.reactions_ec_model = ReactionsECModel()

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
            self.reactions_head = nn.Sequential(
                nn.Linear(1200, 600),
                nn.ReLU(),
                nn.BatchNorm1d(600)
            )

        else:
            self.compounds_head = nn.Sequential(
                nn.Linear(6144, 3072),
                nn.ReLU(),
                nn.Linear(3072, 1536),
                nn.ReLU(),
                nn.Linear(1536, 768),
                nn.ReLU(),
                nn.BatchNorm1d(768),
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.BatchNorm1d(384),
            )

            self.proteins_head = nn.Sequential(
                nn.Linear(1280, 2560),
                nn.ReLU(),
                nn.BatchNorm1d(2560),
                nn.Linear(2560, 5120),
                nn.ReLU(),
                nn.BatchNorm1d(5120),
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
            self.reactions_head = nn.Sequential(
                nn.Linear(2400, 1200),
                nn.BatchNorm1d(1200),
                nn.ReLU(),
                nn.Linear(1200, 600),
                nn.ReLU(),
                nn.BatchNorm1d(600)
            )

        self.esi_model = DNN(1024, [1024, 1024, 512, 256], 1, batch_norm=True, last_sigmoid=True, dropout=None)
        self.reaction_catalysis_model = DNN(1240, [620, 310, 155], 1, batch_norm=True, last_sigmoid=True, dropout=None)


    def forward(self, x_compounds, x_proteins, x_reactions, return_predictions=False):
        if self.transfer_learning:
            with torch.no_grad():
                np_prediction, compounds_embedding = self.compounds_model([x_compounds], return_embedding=True)
                protein_ec_number_prediction, protein_embedding = self.proteins_model([x_proteins], return_embedding=True)
                reactions_ec_prediction, reactions_embedding = self.reactions_ec_model(x_reactions, return_embedding=True)
            
            compounds_embedding = self.compounds_head(compounds_embedding)
            protein_embedding = self.proteins_head(protein_embedding)
            reactions_embedding = self.reactions_head(reactions_embedding)
        else:
            compounds_embedding = self.compounds_head(x_compounds)
            protein_embedding = self.proteins_head(x_proteins)
            reactions_embedding = self.reactions_head(x_reactions)

        # compound_process = self.compounds_model_fp([x_compounds])
        esi_interaction = torch.cat([compounds_embedding, protein_embedding], dim=1)
        esi_interaction = self.esi_model([esi_interaction])

        reaction_catalysis_interaction = torch.cat([protein_embedding, reactions_embedding], dim=1)
        reaction_catalysis_interaction = self.reaction_catalysis_model([reaction_catalysis_interaction])

        if return_predictions:
            return esi_interaction, reaction_catalysis_interaction, np_prediction, protein_ec_number_prediction, reactions_ec_prediction
        else:
            return esi_interaction, reaction_catalysis_interaction
    
class MultiModalModel(L.LightningModule):

    def __init__(self, scheduler=None, protein_model_path="", compounds_model_path="", reactions_model_path="", transfer_learning=True) -> None:
        self.scheduler = scheduler
        self.protein_model_path = protein_model_path
        self.compounds_model_path = compounds_model_path
        self.reactions_model_path = reactions_model_path
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
        
        reactions_ec_model_pretrained_dict_ = torch.load(self.reactions_model_path)
        reactions_ec_model_pretrained_dict = {k.replace("head.3", "reactions_ec_model.0"): v 
                                               for k, v in reactions_ec_model_pretrained_dict_.items() 
                                               if k.replace("head.3", "reactions_ec_model.0") 
                                               in self.interaction_model.reactions_ec_model.state_dict()}
        
        for old_key, new_key in [("head.0", "embedding_part.0"), ("head.1", "embedding_part.1")]:
            reactions_ec_model_pretrained_dict.update({
                k.replace(old_key, new_key): v
                for k, v in reactions_ec_model_pretrained_dict_.items()
                if k.replace(old_key, new_key) in self.interaction_model.reactions_ec_model.state_dict()
            })

        
        self.interaction_model.compounds_model.load_state_dict(np_classifier_model_pretrained_dict)
        self.interaction_model.proteins_model.load_state_dict(torch.load(self.protein_model_path))

        self.interaction_model.reactions_ec_model.load_state_dict(reactions_ec_model_pretrained_dict)


    def _create_model(self):
        self.interaction_model = MultiModalTransferLearning(transfer_learning=self.transfer_learning)
        self.load_models()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=0.001)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x_compounds, x_proteins, x_reactions):
        return self.interaction_model(x_compounds, x_proteins, x_reactions)

    def compute_loss(self, logits_esi, logits_reaction_catalysis, y):

        loss_esi = BCELoss()(logits_esi, y)
        loss_reaction_catalysis = BCELoss()(logits_reaction_catalysis, y)

        return loss_esi + loss_reaction_catalysis
    
    def training_step(self, batch, batch_idx):
        x_proteins, x_compounds, x_reactions, y = batch
        logits_esi, logits_reaction_catalysis = self(x_compounds, x_proteins, x_reactions)
        loss = self.compute_loss(logits_esi, logits_reaction_catalysis, y)
        logits_esi = logits_esi.detach().cpu()
        logits_reaction_catalysis = logits_reaction_catalysis.detach().cpu()
        y = y.detach().cpu()
        
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        
        accuracy = self.accuracy(logits_esi, y)
        f1 = self.f1(logits_esi, y)
        recall = self.recall(logits_esi, y)
        precision = self.precision(logits_esi, y)

        self.log("train_esi_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_esi_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("train_esi_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        self.log("train_esi_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        accuracy = self.accuracy(logits_reaction_catalysis, y)
        f1 = self.f1(logits_reaction_catalysis, y)
        recall = self.recall(logits_reaction_catalysis, y)
        precision = self.precision(logits_reaction_catalysis, y)

        self.log("train_reaction_catalysis_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log("train_reaction_catalysis_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("train_reaction_catalysis_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        self.log("train_reaction_catalysis_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_proteins, x_compounds, x_reactions, target = batch
        output_esi, output_reaction_catalysis = self(x_compounds, x_proteins, x_reactions)

        output_esi = output_esi.detach().cpu()
        output_reaction_catalysis = output_reaction_catalysis.detach().cpu()
        target = target.detach().cpu()

        loss = self.compute_loss(output_esi, output_reaction_catalysis, target)
        accuracy = self.accuracy(output_esi, target)
        f1 = self.f1(output_esi, target)
        recall = self.recall(output_esi, target)
        precision = self.precision(output_esi, target)

        self.log("val_loss", loss.item(), on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_esi_accuracy", accuracy, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_esi_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_esi_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_esi_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        accuracy = self.accuracy(output_reaction_catalysis, target)
        f1 = self.f1(output_reaction_catalysis, target)
        recall = self.recall(output_reaction_catalysis, target)
        precision = self.precision(output_reaction_catalysis, target)

        self.log("val_reaction_catalysis_accuracy", accuracy, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_reaction_catalysis_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_reaction_catalysis_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        self.log("val_reaction_catalysis_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)
        
        return loss
        

    def predict_step(self, batch):
        if len(batch) == 4:
            x_proteins, x_compounds, x_reactions, target = batch
        else:
            x_proteins, x_compounds, x_reactions = batch
        return self(x_compounds, x_proteins, x_reactions)