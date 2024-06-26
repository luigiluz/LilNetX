import torch as ch

from trainer import create_model_and_scaler, update_state_dict
from utils import setup_cuda
from config import Config

MODEL_PATH = "/content/drive/MyDrive/data/checkpoints_15_epochs/cnn_ids_lilnetx_fold0/model_best.pth.tar"

""" Checkpoint sample dictionary
ckpt = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'prob_models': {group_name:prob_model.state_dict() for group_name,prob_model in prob_models.items()},
                'best_acc1': best_acc1,
                'best_epoch': best_epoch,
                'optimizer' : optimizer.state_dict(),
                'prob_optimizer' : prob_optimizer.state_dict(),
                'bits': bits,
                'ac_bytes': ac_bytes
            }
"""

def load_ckpt():
    if conf_ckpt['resume']:
        save_path = os.path.join(conf_ckpt['save_dir'],conf_ckpt['filename'])
        resume_path = save_path if not conf_ckpt['resume_path'] else conf_ckpt['resume_path']
        if os.path.exists(resume_path):
            ckpt = ch.load(resume_path, map_location='cpu')
            start_epoch = ckpt['epoch']
            model.load_state_dict(update_state_dict(model.state_dict(),ckpt['model']))
            for group_name in prob_models:
                prob_models[group_name].load_state_dict(update_state_dict(prob_models[group_name].state_dict(),ckpt['prob_models'][group_name]))
            optimizer.load_state_dict(ckpt['optimizer'])
            prob_optimizer.load_state_dict(ckpt['prob_optimizer'])
            best_acc1 = ckpt['best_acc1']
            best_epoch = ckpt['best_epoch']
            print(f'Checkpoint found, continuing training from epoch {start_epoch}')
            new_seed = conf_common['seed']
            setup_cuda(new_seed,conf_dist['local_rank'])
            print(f'Changing random seed to {new_seed}')
            del ckpt
        else:
            print(f'Resume checkpoint {resume_path} not found, starting training from scratch...')
        epoch_time = time.time()

def main():
    ckpt = ch.load(MODEL_PATH, map_location='cpu')
    model, prob_models, scaler = create_model_and_scaler()

    model.load_state_dict(update_state_dict(model.state_dict(), ckpt['model']))
    for group_name in prob_models:
        prob_models[group_name].load_state_dict(update_state_dict(prob_models[group_name].state_dict(),ckpt['prob_models'][group_name]))



if __name__ == "__main__":
    main()
