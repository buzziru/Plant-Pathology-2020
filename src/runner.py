import os
import math
import glob
import gc
import shutil
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import update_bn
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import BackupHandler, MetricHandler
from .dataset import PlantDataModule, ImageDataset
from .model import PlantDiseaseModule, FoldAlphaCallback

class ExperimentRunner:
    """
    Orchestrates K-Fold Cross Validation and the entire experiment process.
    Handles environment setup, logging, checkpoints, training loops, and inference.
    """
    def __init__(self, config, train_df, teacher_logit, test_df, all_images):
        self.config = config
        self.train_df = train_df
        self.teacher_logit = teacher_logit
        self.test_df = test_df
        self.all_images = all_images
        self.paths = self._setup_env()
        self.backup_handler = BackupHandler(local_dir=self.paths.local_path , backup_dir=self.paths.drive_path, active=False)
        
    def _setup_env(self):
        is_kaggle = os.path.exists('/kaggle/') 
        is_colab = os.path.exists('/content/drive/MyDrive') and not is_kaggle

        if is_kaggle:
            print("Environment: Kaggle")
            drive_path = None
            local_path = '/kaggle/working/'
        elif is_colab:
            print("Environment: Google Colab")
            drive_path = f'/content/drive/MyDrive/Kaggle_Save/{self.config.exp_name}/'
            local_path = '/content/models/'
        else:
            print("Environment: Local")
            drive_path = None
            local_path = f'data/models/{self.config.exp_name}/'
        
        print(f"Save Path: {local_path}")
        return SimpleNamespace(local_path=local_path, drive_path=drive_path)    
    
    def run(self):
        for fold in range(self.config.n_folds):
            print('='*30, f'FOLD {fold+1}', '='*30)
            
            wandb_logger = WandbLogger(
                project=self.config.project_name,
                group=self.config.exp_name,
                name=f"Fold_{fold+1}",
                job_type="train",
                save_code=True,
                config={k: v for k, v in self.config.__dict__.items() if not k.startswith('__')}
            )
            
            train_len = len(self.train_df[self.train_df['fold'] != fold])
            steps_per_epoch = math.ceil(train_len / self.config.batch_size / self.config.accum_iter)

            datamodule = PlantDataModule(
                train_df=self.train_df, 
                teacher_logit=self.teacher_logit, 
                test_df=self.test_df, 
                all_images=self.all_images,
                cfg=self.config, 
                fold_idx=fold
            )
            model = PlantDiseaseModule(self.config, steps_per_epoch=steps_per_epoch)
            
            ckpt_callback = ModelCheckpoint(
                monitor='val_roc_auc',
                mode='max',
                save_top_k=self.config.top_k,
                save_weights_only=True,
                save_last=False,
                dirpath=self.paths.local_path,
                filename=f'Fold{fold+1}-Ep{{epoch:02d}}-{{val_roc_auc:.4f}}',
                auto_insert_metric_name=False,
            )
            progress_bar = TQDMProgressBar(refresh_rate=1)
            alpha_callback = FoldAlphaCallback(
                current_fold=fold,
                weak_folds=[0, 1],
                weak_alpha=self.config.weak_alpha, 
                strong_alpha=self.config.strong_alpha
            )
            
            trainer = pl.Trainer(
                max_epochs=self.config.epochs,
                accelerator='auto',
                precision='16-mixed',
                accumulate_grad_batches=self.config.accum_iter,
                callbacks=[ckpt_callback, progress_bar, alpha_callback],
                logger=wandb_logger,
                log_every_n_steps=10
            )

            trainer.fit(model, datamodule=datamodule)
            
            print(f'\\n Top-{ckpt_callback.save_top_k} Models in this Fold:')
            for path, score in ckpt_callback.best_k_models.items():
                model_name = os.path.basename(path)
                print(f'> {model_name}')
                
            wandb.finish()
            
            # Memory Cleanup
            del datamodule, trainer, model
            torch.cuda.empty_cache()
            gc.collect()

    def _load_averaged_model(self, fold):
        save_path = os.path.join(self.paths.local_path, f'best_score_model_{fold+1}.pth')
        model = PlantDiseaseModule(self.config)

        if os.path.exists(save_path):
            print(f'Found existing averaged model for Fold {fold+1}. Loading directly...')
            state_dict = torch.load(save_path, map_location=self.config.device)
            model.load_state_dict(state_dict)
        else:
            print(f'Merging Top-K Models for Fold {fold+1} ...')
            score_pattern = os.path.join(self.paths.local_path, f'Fold{fold+1}-Ep*.ckpt')
            score_files = glob.glob(score_pattern)
            print(f'Found {len(score_files)} score models : {[os.path.basename(f) for f in score_files]}')
            
            if not score_files:
                raise FileNotFoundError(f"No checkpoint files found for Fold {fold+1}")

            first_state = torch.load(score_files[0], map_location='cpu')['state_dict']
            avg_state_dict = {}
            for k, v in first_state.items():
                if v.is_floating_point():
                    avg_state_dict[k] = v.float() # Convert to Float32 for averaging
                else:
                    avg_state_dict[k] = v 
            
            if len(score_files) > 1:
                for path in score_files[1:]:
                    state_dict = torch.load(path, map_location='cpu')['state_dict']
                    for key in avg_state_dict:
                        if avg_state_dict[key].is_floating_point():
                            avg_state_dict[key] += state_dict[key].float()
                        else:
                            pass
                for key in avg_state_dict:
                    if avg_state_dict[key].is_floating_point():
                        avg_state_dict[key] = avg_state_dict[key] / len(score_files)
            
            model.load_state_dict(avg_state_dict)
            
            for remove_path in score_files:
                if os.path.exists(remove_path):
                    os.remove(remove_path)
            
            # Save averaged model without BN update first (in case BN update fails or is skipped)
            # But here we stick to the flow
            
        # BN Update
        if self.config.is_bn:
            print('Update BN stats ... ')
            model = model.to(self.config.device)
            model.train()

            train_subset = self.train_df[self.train_df['fold'] != fold].reset_index(drop=True)
            transform_test = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            # Pass self.all_images here
            dataset_bn = ImageDataset(train_subset, self.all_images, transform=transform_test, is_test=True)
            loader_bn = DataLoader(
                dataset_bn, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            update_bn(loader_bn, model, device=self.config.device)
            model.eval()
            torch.save(model.state_dict(), save_path)
            print('Save Avg Model (with BN updated) : ', save_path)
        else:
            print('Skipping BN update for LayerNorm')
            model = model.to(self.config.device)
            torch.save(model.state_dict(), save_path) # Save if no BN update
            
        return model

    def find_optimal_temperature(self, logits, labels):
        """
        Find T that minimizes NLL.
        """
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # NLL Loss search
        t_candidates = np.arange(0.5, 2.6, 0.1)
        best_t = min(t_candidates, key=lambda t: torch.nn.CrossEntropyLoss()(logits_tensor / t, labels_tensor).item())
        print(f"    > Best T: {best_t:.1f}")
        return best_t

    def run_inference(self):
        oof_preds = np.zeros((len(self.train_df), 4))
        oof_preds_og = np.zeros((len(self.train_df), 4))
        oof_preds_logit = np.zeros((len(self.train_df), 4))

        final_preds = np.zeros((len(self.test_df), 4))
        final_preds_og = np.zeros((len(self.test_df), 4))

        for fold in range(self.config.n_folds):
            print(f'=== Inference Fold {fold+1} ===')
            avg_model = self._load_averaged_model(fold)

            # Inference DataModule
            infer_module = PlantDataModule(
                train_df=self.train_df, 
                teacher_logit=self.teacher_logit, 
                test_df=self.test_df, 
                all_images=self.all_images,
                cfg=self.config, 
                fold_idx=fold, 
                inference_mode=True
            )
            infer_module.setup(stage='test')
            
            progress_bar = TQDMProgressBar(refresh_rate=1)
            infer_trainer = pl.Trainer(
                accelerator='auto',
                precision='16-mixed',
                logger=False,
                enable_checkpointing=False,
                callbacks=[progress_bar]
            )

            # Validation Indices and Labels
            valid_indices = self.train_df[self.train_df['fold'] == fold].index.values
            valid_labels = self.train_df.iloc[valid_indices][['healthy', 'multiple_diseases', 'rust', 'scab']].values

            # OOF Inference
            oof_list = infer_trainer.predict(avg_model, dataloaders=infer_module.val_dataloader())
            current_oof_logits = torch.cat(oof_list).cpu().numpy()
            print(f"Max: {current_oof_logits.max()}, Min: {current_oof_logits.min()}")

            # Calibration
            optimal_t = self.find_optimal_temperature(current_oof_logits, valid_labels)

            # OOF Calibration
            calibrated_oof_probs = torch.softmax(torch.tensor(current_oof_logits) / optimal_t, dim=1).numpy()
            og_oof_probs = torch.softmax(torch.tensor(current_oof_logits), dim=1).numpy()
            oof_preds[valid_indices] = calibrated_oof_probs
            oof_preds_og[valid_indices] = og_oof_probs
            oof_preds_logit[valid_indices] = current_oof_logits

            # Test Inference
            sub_list = infer_trainer.predict(avg_model, dataloaders=infer_module.predict_dataloader())
            current_test_logits = torch.cat(sub_list).cpu().numpy()
            calibrated_test_probs = torch.softmax(torch.tensor(current_test_logits) / optimal_t, dim=1).numpy()
            test_probs = torch.softmax(torch.tensor(current_test_logits), dim=1).numpy()
            final_preds += calibrated_test_probs
            final_preds_og += test_probs

            # Cleanup
            del avg_model, infer_trainer, infer_module
            torch.cuda.empty_cache()
            gc.collect()

        final_preds /= self.config.n_folds
        final_preds_og /= self.config.n_folds

        # Final Metrics
        metric_handler = MetricHandler()
        metric_handler.update(oof_preds, self.train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].values)
        oof_roc = metric_handler.compute_roc_auc()
        print(f'\\n>>> Final OOF ROC AUC : {oof_roc:.5f}')

        return oof_preds, oof_preds_og, oof_preds_logit, final_preds, final_preds_og
