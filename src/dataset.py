import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from .utils import seed_worker

# Global generator for reproducibility
g = torch.Generator()

class MetadataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def prepare_data(self):
        train_df = pd.read_csv(self.cfg.train_csv)
        train_df = train_df.reset_index(drop=True)
        test_df = pd.read_csv(self.cfg.test_csv)
        submission = pd.read_csv(self.cfg.submission_csv)
        
        # Load Teacher Logits
        teacher_logit1 = np.load(self.cfg.logit_path1)
        teacher_logit2 = np.load(self.cfg.logit_path2)
        teacher_logit = teacher_logit1 * 0.5 + teacher_logit2 * 0.5
        
        print(f"Logit1 Range: {teacher_logit1.min():.2f} ~ {teacher_logit1.max():.2f}")
        print(f"Logit2 Range: {teacher_logit2.min():.2f} ~ {teacher_logit2.max():.2f}")
        
        return train_df, teacher_logit, test_df, submission

def load_all_images(img_dir, image_ids):
    """
    Loads all images into RAM as done in the notebook.
    """
    all_images = {}
    print(f"Loading images from {img_dir} into RAM...")
    
    # Check if directory exists
    if not os.path.exists(img_dir):
         # Fallback for notebook compatibility if paths differ, but CFG should handle this.
         # If running from root, data/images/ is expected.
         pass

    for img_id in tqdm(image_ids, desc='Loading Images...', leave=False):
        path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(path):
            continue # process error or skip
            
        img = cv2.imread(path)
        img = cv2.resize(img, (650, 450))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.setflags(write=False) # Make read-only for sharing
        all_images[img_id] = img
        
    print(f"Loaded {len(all_images)} images into RAM")
    return all_images

class ImageDataset(Dataset):
    def __init__(self, df, all_images, hard_cols=['healthy', 'multiple_diseases', 'rust', 'scab'], teacher_logit=None, transform=None, is_test=False):
        super().__init__()
        self.df = df
        self.all_images = all_images
        self.transform = transform
        self.is_test = is_test
        self.hard_cols = hard_cols
        self.teacher_logit = teacher_logit

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0] # Assuming image_id is first column
        
        # Safety check if image exists in RAM
        if img_id not in self.all_images:
             raise FileNotFoundError(f"Image {img_id} not found in loaded images.")

        image = self.all_images[img_id].copy()

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image
        else:
            hard_labels = self.df.iloc[idx][self.hard_cols].values.astype(np.float32)
            oof_logit = self.teacher_logit[idx]
            return image, torch.tensor(oof_logit), torch.tensor(hard_labels)

class PlantDataModule(pl.LightningDataModule):
    def __init__(self, train_df, teacher_logit, test_df, all_images, cfg, fold_idx, inference_mode=False):
        super().__init__()
        self.train_df = train_df
        self.teacher_logit = teacher_logit
        self.test_df = test_df
        self.all_images = all_images # Shared dict
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.inference_mode = inference_mode
        g.manual_seed(cfg.seed)

        self.transform_train = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0)
            ], p=1.0),

            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.5),
            
            A.OneOf([
                A.Affine(
                    scale=(0.8, 1.2),       
                    translate_percent=0.2,  
                    rotate=45,              
                    shear=20,               
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT_101, 
                    p=1.0
                ),
                A.Perspective(scale=(0.05, 0.1), p=1.0), 
            ], p=0.8),
            
            A.OneOf([
                A.ISONoise(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.3),

            A.CoarseDropout(
                num_holes_range=(8, 16),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=[103, 131, 82],
                p=0.5
            ),

            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.transform_test = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_mask = (self.train_df['fold']!=self.fold_idx).values
            val_mask = (self.train_df['fold']==self.fold_idx).values

            self.train = self.train_df[train_mask].reset_index(drop=True)
            self.valid = self.train_df[val_mask].reset_index(drop=True)
            train_logit = self.teacher_logit[train_mask]
            val_logit = self.teacher_logit[val_mask]

            self.dataset_train = ImageDataset(self.train, self.all_images, teacher_logit=train_logit, transform=self.transform_train)
            self.dataset_valid = ImageDataset(self.valid, self.all_images, teacher_logit=val_logit, transform=self.transform_test)
            print(f'[Fit] Train: {len(self.train)}, Valid: {len(self.valid)}')

        elif stage == 'test':
            val_mask = (self.train_df['fold'] == self.fold_idx).values
            self.valid = self.train_df[val_mask].reset_index(drop=True)
            val_logit = self.teacher_logit[val_mask]
            
            self.dataset_valid = ImageDataset(self.valid, self.all_images, teacher_logit=val_logit, transform=self.transform_test)
            self.dataset_test = ImageDataset(self.test_df, self.all_images, transform=self.transform_test, is_test=True)
            print(f'[Test] Valid(OOF): {len(self.valid)}, Test: {len(self.test_df)}')

        elif stage == 'predict':
            self.dataset_test = ImageDataset(self.test_df, self.all_images, transform=self.transform_test, is_test=True)
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, shuffle=True,
                          worker_init_fn=seed_worker, generator=g, num_workers=self.cfg.num_workers, 
                          persistent_workers=True, pin_memory=True)
    
    def val_dataloader(self):
        user_persistent = not self.inference_mode
        return DataLoader(self.dataset_valid, batch_size=self.cfg.batch_size*4, shuffle=False,
                          worker_init_fn=seed_worker, generator=g, num_workers=self.cfg.num_workers, 
                          persistent_workers=user_persistent, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.batch_size*4, shuffle=False,
                          worker_init_fn=seed_worker, generator=g, num_workers=self.cfg.num_workers, 
                          persistent_workers=False, pin_memory=True)
    
    def test_dataloader(self):
        return self.predict_dataloader()
