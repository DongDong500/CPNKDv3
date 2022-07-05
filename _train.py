import os

from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter


from _get_dataset import _get_dataset
from _validate import _validate
from _load_model import _load_model

def _train(opts, devices, run_id) -> dict:

    logdir = os.path.join(opts.Tlog_dir, 'run_' + str(run_id).zfill(2))
    writer = SummaryWriter(log_dir=logdir) 

    train_dst, val_dst = _get_dataset(opts)
    
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    