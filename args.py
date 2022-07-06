import os
import socket
import argparse
from datetime import datetime

import network
import datasets
import utils

HOSTNAME = {
    "server2" : 2,
    "server3" : 3,
    "server4" : 4,
    "server5" : 5
}
LOGIN = {
    "server2" : "/mnt/server5/sdi/login.json",
    "server3" : "/mnt/server5/sdi/login.json",
    "server4" : "/mnt/server5/sdi/login.json",
    "server5" : "/data1/sdi/login.json"
}
DEFAULT_PREFIX = {
    "server2" : "/mnt/server5/sdi",
    "server3" : "/mnt/server5/sdi",
    "server4" : "/mnt/server5/sdi",
    "server5" : "/data1/sdi"
}
DATA_DIR = {
    "server2" : "/mnt/server5/sdi/datasets",
    "server3" : "/mnt/server5/sdi/datasets",
    "server4" : "/mnt/server5/sdi/datasets",
    "server5" : "/data1/sdi/datasets"
}

def _get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--default_prefix", type=str, default='/',
                        help="path to results (default: /")
    parser.add_argument("--data_root", type=str, default='/',
                        help="path to Dataset (default: /")
    parser.add_argument("--current_time", type=str, default="current time",
                        help="results images folder name (default: current time)")
    # Log-in info
    parser.add_argument("--login_dir", type=str, default='/',
                        help="path to user log-in info json file (default: /)")
    parser.add_argument("--cur_work_server", type=int, default=0,
                        help="current working server (default: 0)")
    # Tensorboard options
    parser.add_argument("--save_Tlog", action='store_true', default=False, 
                        help="save log to default path (default: False)")
    parser.add_argument("--Tlog_dir", type=str, default='/',
                        help="path to tensorboard log (default: /)")
    
    # Model options
    available_models = sorted(name for name in network.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.model.__dict__[name]) )
    parser.add_argument("--t_model", type=str, default='unet_rgb', choices=available_models,
                        help='model name (default: Unet RGB)')
    parser.add_argument("--s_model", type=str, default='unet_rgb', choices=available_models,
                        help='model name (default: Unet RGB)')
    parser.add_argument("--t_model_params", type=str, default='/',
                        help="pretrained teacher network params (default: '/')")
    # DeeplabV3+ options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32, 64],
                        help="output stride (default: 16)")
    parser.add_argument("--t_separable_conv", action='store_true', default=False,
                        help="teacher model: apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--t_output_stride", type=int, default=16, choices=[8, 16, 32, 64],
                        help="teacher model: output stride (default: 16)")
    # Resume model from checkpoint
    parser.add_argument("--resume_ckpt", default='/', type=str,
                        help="resume from checkpoint (defalut: /)")
    parser.add_argument("--continue_training", action='store_true', default=False,
                        help="restore state from reserved params (defaults: false)")

    # Dataset options
    parser.add_argument("--num_workers", type=int, default=12,
                        help="number of workers (default: 12)")
    available_datasets = sorted( name for name in datasets.getdata.__dict__ if  callable(datasets.getdata.__dict__[name]) )
    parser.add_argument("--dataset", type=str, default="CPN_six", choices=available_datasets,
                        help='Name of dataset (default: CPN_six)')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes (default: 2)")
    parser.add_argument("--is_rgb", action='store_false', default=True,
                        help="choose True: RGB, False: gray (default: True)")
    parser.add_argument("--dataset_ver", type=str, default="splits/v5/3",
                        help="version of dataset (default: splits/v5/3)")
    parser.add_argument("--tvs", type=int, default=5,
                        help="number of blocks to split train set (default: 5)")

    # Augmentation options
    parser.add_argument("--resize", default=(496, 468))
    parser.add_argument("--crop_size", default=(512, 448))
    parser.add_argument("--val_crop_size", default=(512, 448))
    parser.add_argument("--scale_factor", type=float, default=5e-1)

    # Train options
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--gpus", type=str, default='6,7',
                        help="GPU IDs (default: 6,7)")
    parser.add_argument("--total_itrs", type=int, default=2500,
                        help="epoch number (default: 2.5k)")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--loss_type", type=str, default='dice_loss',
                        help="criterion (default: dice loss)")
    parser.add_argument("--optim", type=str, default='SGD',
                        help="optimizer (default: SGD)")
    parser.add_argument("--lr_policy", type=str, default='step',
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=100, 
                        help="step size (default: 1000)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--exp_itr", type=int, default=2,
                        help='repeat N times identical experiments (default: 2)')
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="alpha for KD loss (default: 0.5)")
    parser.add_argument("--T", type=float, default=3,
                        help="temperature in KD loss (default: 3)")
    parser.add_argument("--std", type=float, default=0.01,
                        help="[train] sigma in gaussian perturbation (default: 0.01)")

    # Early stop options
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no improvement after which training will be stopped (default: 100)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (default: 0.001)")

    # Validate options
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validate (default: 4)')
    parser.add_argument("--val_std", type=float, default=0.01,
                        help="[val] sigma in gaussian perturbation (default: 0.01)")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help='save validate segmentation results to \"./val\" (default: False)')
    parser.add_argument("--val_results_dir", type=str, default='/',
                        help="save segmentation results to (default: /)")

    # Test options
    parser.add_argument("--test_interval", type=int, default=1,
                        help="epoch interval for test (default: 1)")
    parser.add_argument("--test_batch_size", type=int, default=4,
                        help='batch size for test (default: 4)')
    parser.add_argument("--save_test_results", action='store_true', default=False,
                        help='save test results to \"./test\" (default: False)')
    parser.add_argument("--test_results_dir", type=str, default='/',
                        help="save segmentation results to (default: /)")

    # Outcome options
    parser.add_argument("--save_model", action='store_true', default=False,
                        help="save best model param to \"./best-param\" (default: False)")
    parser.add_argument("--best_ckpt", type=str, default=None,
                        help="save best model param to \"./best-param\"")
    
    # Run Demo
    parser.add_argument("--run_demo", action='store_true', default=False)

    return parser


def get_argparser():

    if socket.gethostname() in HOSTNAME.keys():
        parser = _get_argparser().parse_args()

        save_folder = os.path.dirname( os.path.abspath(__file__) ).split('/')[-1] + '-result'

        parser.default_prefix = os.path.join(DEFAULT_PREFIX[socket.gethostname()], save_folder)
        parser.data_root = DATA_DIR[socket.gethostname()]
        parser.current_time = datetime.now().strftime('%b%d_%H-%M-%S') + ('_demo' if parser.run_demo else '')
        parser.login_dir = LOGIN[socket.gethostname()]
        parser.cur_work_server = socket.gethostname()

        if not os.path.exists(parser.login_dir):
            raise FileNotFoundError ("Log-In file not found or corrupted. " +
                                     parser.login_dir)
        if not os.path.exists(parser.data_root):
            raise RuntimeError('Dataset root directory not found or corrupted. \n' +
                               parser.data_root)

        _dir = os.path.join(DEFAULT_PREFIX[socket.gethostname()], save_folder, parser.current_time)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

        if parser.save_Tlog:
            parser.Tlog_dir = os.path.join(_dir, 'log')
            os.mkdir(parser.Tlog_dir)

        if parser.save_val_results:
            parser.val_results_dir = os.path.join(_dir, 'val')
            os.mkdir(parser.val_results_dir)

        if parser.save_test_results:
            parser.test_results_dir = os.path.join(_dir, 'test')
            os.mkdir(parser.test_results_dir)

        if parser.save_model:
            parser.best_ckpt = os.path.join(_dir, 'best-param')
            os.mkdir(parser.best_ckpt)
        else:
            parser.best_ckpt = os.path.join(_dir, 'cache-param')
            os.mkdir(parser.best_ckpt)
        
        return parser
    else:
        raise NotImplementedError ("Host name not matched: ", socket.gethostname())


def save_argparser(parser, save_dir) -> dict:

    jsummary = {}
    for key, val in vars(parser).items():
        jsummary[key] = val

    utils.save_dict_to_json(jsummary, os.path.join(save_dir, 'summary.json'))

    return jsummary



if __name__ == "__main__":

    import utils

    print('basename:    ', os.path.basename(__file__)) # main.py
    print('dirname:     ', os.path.dirname(__file__)) # /data1/sdi/CPNKD
    print('abspath:     ', os.path.abspath(__file__)) # /data1/sdi/CPNKD/main.py
    print('abs dirname: ', os.path.dirname(os.path.abspath(__file__))) # /data1/sdi/CPNKD

    opts = get_argparser()
    jsummary = {}
    for key, val in vars(opts).items():
        jsummary[key] = val
    utils.save_dict_to_json(d=jsummary, json_path='/mnt/server5/sdi/CPNKDv3/utils/sample/opts_sample.json')

    pram = utils.Params('/mnt/server5/sdi/CPNKDv3/utils/sample/opts_sample.json')

    print(type(pram.separable_conv)) # bool
    print(type(pram.num_classes)) # int
    print(type(pram.weight_decay)) # float
    print(type(pram.best_ckpt)) # str

    pram.update(json_path='/mnt/server5/sdi/CPNKDv3/utils/sample/mlog.json')
    utils.save_dict_to_json(pram.__dict__, '/mnt/server5/sdi/CPNKDv3/utils/sample/out.json')