import torch

class CFG:
    model_arch = 'resnest101e'
    is_bn = True
    seed = 938
    lr = 0.0001
    weight_decay = 0.05
    alpha = 0.7
    weak_alpha = 0.3
    strong_alpha = 0.7
    T = 1.25
    drop_path_rate = 0.2
    top_k = 3
    n_folds = 5
    epochs = 24
    batch_size = 32
    accum_iter = 1
    num_workers = 4
    persistent_workers = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = 'PlantPathology2020'
    exp_name = 's14_resnest'
    
    # Path configurations
    data_root = 'data/'
    image_dir = 'data/images/'
    train_csv = 'data/datasets/train_reborn_02.csv'
    test_csv = 'data/test.csv'
    submission_csv = 'data/sample_submission.csv'
    # Logit paths (Update these paths as needed based on your file structure)
    logit_path1 = 'data/models/s10_convnext_small_T_scheduler/oof_ogit_s10_convnext_small_T_scheduler.npy'
    logit_path2 = 'data/models/s12_conv_teacher_s10_v2/oof_logit_s12_conv_teacher_s10_v2.npy'
