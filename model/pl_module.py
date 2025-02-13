import lightning as L
import torch
from typing import Optional
import torch.optim
from lightning import LightningModule
from model.models import GPT
from omegaconf import DictConfig
import numpy as np
from model.diffusion_utils import CategoricalDiffusion, InferenceSchedule, prepare_diffusion


class Pl_model_wrapper(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        train_config: DictConfig
    ):
        super().__init__()
        self.betas = train_config.betas
        self.weight_decay = train_config.weight_decay
        self.learning_rate = train_config.learning_rate
        self.patience = train_config.patience
        self.model_config = model_config
        self.num_vars = model_config.num_vars
        self.save_hyperparameters()
        self.gpt = GPT(model_config)
        self.diffusion = CategoricalDiffusion(T=1000, schedule='linear')

    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gpt(inputs)

    def configure_optimizers(self) -> dict:
        optimizer = self.gpt.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=self.betas
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1.e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        t, xt = prepare_diffusion(batch, self.diffusion,self.num_vars)
        logits = self.gpt(batch,xt, t)
        var_preds = logits[:,:self.num_vars,:].reshape(-1,2)
        node_labels =((batch['labels'] + 1)/2).reshape(-1)
        loss = torch.nn.functional.cross_entropy(var_preds, node_labels.long().to(var_preds.device)) 
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ret = self.validation_diffusion(batch)
        return ret
        # logits, loss = self(inputs)
        # labels = inputs['labels'][:,:,:self.num_vars]
        # logits = logits[:,:,:self.num_vars]
        # predictions = (torch.sigmoid(logits) >= 0.5)
        # predictions = predictions.to(torch.int64)
    

        # correct_samples = torch.all(predictions == labels, dim=1)
        # accuracy = correct_samples.float().mean().item()

        # #accuracy = self.calculate_accuracy(logits, targets)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)    
        # return {'val_loss': loss}

    def calculate_accuracy(self, logits, targets):
        pass
        #predictions = torch.argmax(logits, dim=-1)
        #targets = targets[:, 0] # prev_enabling, enabling, subgoal, prev_start, prev_subgoal
        #return (predictions == targets).float().mean()
    def categorical_denoise_step(self, xt, t, device, batch, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.gpt(
                batch,
                xt.float(),
                t.float(),
            )
            xt = xt.to(device)
            t = t.to(device)
            num_vars = xt.shape[-1]
            x0_pred = x0_pred[:,:num_vars] # batch size is 1 so it takes the first element
            x0_pred_prob = x0_pred.softmax(dim=-1)
            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt
        
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        
        """Sample from the categorical posterior for a given time step.
            See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion
    
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)
    
        # Thanks to Daniyar and Shengyu, who found the "target_t == 0" branch is not needed :)
        # if target_t > 0:
        Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        # else:
        #   Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)
    
        xt = torch.nn.functional.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)
    
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)
    
        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3
    
        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)
    
        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new
    
        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]
    
        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)
    
        """
        if self.sparse:
            xt = xt.reshape(-1)
        """
        return xt
    
    def validation_diffusion(self, batch):
        device = batch['labels'].device
        batch_size = 1
        steps = 25

        node_labels = (batch['labels'].cpu() + 1)/2
        xt = torch.randn_like(node_labels.float())
        xt = (xt > 0).long()
        xt = xt.reshape(-1).unsqueeze(0)
        #xt = xt * 2 - 1

        time_schedule = InferenceSchedule(inference_schedule="cosine",
                                        T=self.diffusion.T, inference_T=steps)
        for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
            t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

            xt = self.categorical_denoise_step(xt, t1, device, batch, target_t=t2)
            #xt = xt.squeeze(2)

        predict_labels = xt.float().cpu().detach().numpy() + 1e-6     

        infered_assignment = np.round(predict_labels)
        assert not np.any((infered_assignment !=0)&(infered_assignment !=1)) # check if all values are 0 or 1
        infered_assignment = infered_assignment * 2 - 1  # convert to -1,1

        result_literals = []
        clauses = batch['pos_embeddings'][0,:,2*self.num_vars:].T - self.num_vars - 1 # recenter back
        for ix, assignment in enumerate(list(infered_assignment[0])):
            result_literals.append(int((ix+1) * assignment))
        
        sat_num = 0
        for c in clauses:
            for lit in c:
                if lit == -self.num_vars-1: # TODO: fix this -1 is padding token
                    break
                if lit in result_literals:
                    sat_num +=1
                    break
        
        gap = len(clauses)-sat_num
        if gap == 0:
            acc = 1
        else:
            acc = 0

        
        self.log('val_avg_gap', gap, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, prog_bar=True, logger=True)

        return 1
                