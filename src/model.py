import torch
import lightning.pytorch as pl
import timm
import ttach as tta
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import ConfusionMatrix, AUROC

class FoldAlphaCallback(Callback):
    """
    Adjusts Alpha for Knowledge Distillation per fold.
    """
    def __init__(self, current_fold, weak_folds=[0, 1], weak_alpha=0.8, strong_alpha=0.5):
        super().__init__()
        self.current_fold = current_fold
        self.weak_folds = weak_folds
        self.weak_alpha = weak_alpha
        self.strong_alpha = strong_alpha

    def on_train_start(self, trainer, pl_module):
        if self.current_fold in self.weak_folds:
            pl_module.current_alpha = self.weak_alpha
            strategy = "GT Focus (Weak Fold)"
        else:
            pl_module.current_alpha = self.strong_alpha
            strategy = "Regularization (Strong Fold)"
            
        if trainer.global_rank == 0:
            print(f"\n[FoldAlphaCallback] Fold {self.current_fold+1}: "
                  f"Alpha set to {pl_module.current_alpha} ({strategy})")

class PlantDiseaseModule(pl.LightningModule):
    def __init__(self, config, steps_per_epoch=None):
        super().__init__()
        # If config is a class, convert to dict
        if isinstance(config, type):
            config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        else:
            config_dict = config
            
        self.save_hyperparameters(config_dict)
        self.current_alpha = self.hparams.alpha
        
        # Load model
        self.model = timm.create_model(
            self.hparams.model_arch,
            pretrained=True,
            drop_path_rate=self.hparams.drop_path_rate,
            num_classes=4
            )
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # TTA
        self.tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ])
        
        # Metrics
        self.valid_auc = AUROC(task='multiclass', num_classes=4)
        self.valid_cm = ConfusionMatrix(task='multiclass', num_classes=4)
        self.best_score = 0.0

        self.top_k_scores = []
        self.top_k = self.hparams.top_k

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)
        
        scheduler_config = {
            'scheduler' : scheduler,
            'interval' : 'epoch',
            'frequency' : 1
        }
        return [optimizer], [scheduler_config]
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        T = self.hparams.T
        alpha = self.current_alpha

        image, logit_from_oof, hard_labels = batch
        outputs = self.model(image)

        # Hard loss
        loss_hard = self.criterion(outputs, hard_labels)

        # Distillation (Soft) loss
        teacher_probs = torch.softmax(logit_from_oof / T, dim=1)
        student_log_probs = torch.nn.functional.log_softmax(outputs / T, dim=1)

        kl_loss = torch.nn.functional.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        )
        loss_soft = kl_loss * (T**2)
        
        # Total loss
        loss = alpha * loss_hard + (1 - alpha) * loss_soft
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('hard_loss', loss_hard, on_step=False, on_epoch=True, logger=True)
        self.log('soft_loss', loss_soft, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, _, hard_labels = batch
        outputs = self.model(image)
        loss = self.criterion(outputs, hard_labels)
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        targets = torch.argmax(hard_labels, dim=1)

        self.valid_cm(preds, targets)
        self.valid_auc(probs, targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_roc_auc', self.valid_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
                  
        score = self.trainer.callback_metrics.get('val_roc_auc')
        current_epoch = self.current_epoch
        
        if score is not None:
            current_score = score.item()            
            self.top_k_scores.append((current_score, current_epoch))
            self.top_k_scores.sort(key=lambda x: x[0], reverse=True)
            self.top_k_scores = self.top_k_scores[:self.top_k]
            is_in_top_k = (current_score, current_epoch) in self.top_k_scores
            
            if is_in_top_k and isinstance(self.logger, WandbLogger):
                # Log Confusion Matrix to WandB
                cm = self.valid_cm.compute().cpu().numpy()
                columns = ['Healthy', 'Multiple', 'Rust', 'Scab']
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=columns, yticklabels=columns,
                            annot_kws={"size": 14})
                            
                plt.ylabel('True Label', fontsize=12)
                plt.xlabel('Predicted Label', fontsize=12)
                plt.title(f'Confusion Matrix (Epoch {current_epoch})', fontsize=14)
             
                self.logger.experiment.log({
                    "val/confusion_matrix": wandb.Image(plt, caption=f"Epoch {current_epoch}"),
                    "global_step": self.global_step
                })
                plt.close()

        self.valid_cm.reset()

    def on_train_epoch_end(self):
        score = self.trainer.callback_metrics.get('val_roc_auc')
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        val_loss = self.trainer.callback_metrics.get('val_loss')

        current_epoch = self.current_epoch
        t_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        v_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        roc_str = f"{score:.4f}" if score is not None else "N/A"
        if self.global_rank == 0:
            print(f"\n(Epoch {current_epoch}) Train Loss: {t_loss_str} | Val Loss: {v_loss_str} | ROC AUC: {roc_str}")        

    def on_predict_start(self):
        self.tta_model = tta.ClassificationTTAWrapper(
            self.model, 
            self.tta_transforms, 
            merge_mode='mean'
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        outputs = self.tta_model(x)
        return outputs
