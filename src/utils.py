import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def set_seed(seed, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    torch.cuda.manual_seed_all(seed) # multi-gpu
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MetricHandler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds_list = []
        self.actual_list = []

    def update(self, preds, actual):
        self.preds_list.extend(preds)
        self.actual_list.extend(actual)

    def compute_roc_auc(self):
        return roc_auc_score(self.actual_list, self.preds_list)

class BackupHandler:
    def __init__(self, local_dir, backup_dir=None, active=True):
        self.local_dir = local_dir
        self.backup_dir = backup_dir
        self.active = active and (backup_dir is not None)

        if self.active and self.backup_dir is not None:
            os.makedirs(self.backup_dir, exist_ok=True)
            print(f'Backup Active : {self.local_dir} -> {self.backup_dir}')

    def backup(self, filename):
        if not self.active or self.backup_dir is None:
            return

        src_path = os.path.join(self.local_dir, filename)
        dst_path = os.path.join(self.backup_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    def save_file(self, data, filename, logit=False):
        local_path = os.path.join(self.local_dir, filename)
        os.makedirs(self.local_dir, exist_ok=True) # Ensure directory exists

        if logit:
            np.save(local_path, data)
            print(f'Logit saved at {local_path}')
        else:
            data.to_csv(local_path, index=False)
            print(f'CSV saved at {local_path}')

        self.backup(filename)
