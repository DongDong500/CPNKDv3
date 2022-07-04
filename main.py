import os
import json
from datetime import datetime

import torch

from utils import MailSend
from kdTrain import train
from args import get_argparser, save_argparser
import utils


if __name__ == '__main__':

    opts = get_argparser()
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (device, str(opts.gpus)))

    mlog = {}
    total_time = datetime.now()
    try:
        opts.loss_type = 'kd_loss'
        opts.s_model = 'deeplabv3plus_resnet50'
        opts.t_model = 'deeplabv3plus_resnet50'
        opts.t_model_params = '/mnt/server5/sdi/CPNnetV1-result/deeplabv3plus_resnet50/May17_07-37-30_CPN_six/best_param/dicecheckpoint.pt'
        opts.output_stride = 32
        opts.t_output_stride = 32

        params = save_argparser(opts, os.path.join(opts.default_prefix, opts.current_time))

        start_time = datetime.now()

        #mlog['Single experimnet'] = train(devices=device, opts=opts)
        
        mlog['Single experimnet'] = {
                                        'Model' : opts.s_model, 'Dataset' : opts.dataset,
                                        'Policy' : opts.lr_policy, 'OS' : str(opts.output_stride),
                                        'F1 [0]' : "{:.6f}".format(0.85260014), 'F1 [1]' : "{:.6f}".format(0.85260014)
                                    }
        params['Single experimnet'] = mlog['Single experimnet']

        time_elapsed = datetime.now() - start_time

        mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
        
        with open(os.path.join(opts.default_prefix, opts.current_time, 'mlog.json'), "w") as f:
            ''' JSON treats keys as strings
            '''
            json.dump(mlog, f, indent=4)
        
        params["time_elpased"] = str(time_elapsed)

        utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))
 
        # Transfer results by G-mail
        MailSend(subject = "Short report-%s" % "CPN Knowledge distillation", 
                    msg = mlog,
                    login_dir = opts.login_dir,
                    ID = 'singkuserver',
                    to_addr = ['sdimivy014@korea.ac.kr']).send()

        os.remove(os.path.join(opts.default_prefix, opts.current_time, 'mlog.json'))

    except KeyboardInterrupt:
        print("Stop !!!")

    except Exception as e:
        print("Error", e)

    total_time = datetime.now() - total_time

    print('Time elapsed (h:m:s.ms) {}'.format(total_time))