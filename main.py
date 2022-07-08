import os
import traceback
from datetime import datetime

from utils import MailSend
from kdTrain import train
from args import get_argparser, save_argparser
import utils


if __name__ == '__main__':

    mlog = {}
    opts = get_argparser()
    total_time = datetime.now()
    
    try:
        opts.loss_type = 'kd_loss'
        opts.s_model = 'deeplabv3plus_resnet50'
        opts.t_model = 'deeplabv3plus_resnet50'
        opts.t_model_params = '/data1/sdi/CPNnetV1-result/deeplabv3plus_resnet50/May17_07-37-30_CPN_six/best_param/dicecheckpoint.pt'
        opts.output_stride = 32
        opts.t_output_stride = 32

        params = save_argparser(opts, os.path.join(opts.default_prefix, opts.current_time))

        start_time = datetime.now()

        mlog['Single experimnet'] = train(opts)
        params['Single experimnet'] = mlog['Single experimnet']

        time_elapsed = datetime.now() - start_time

        mlog['time elapsed'] = 'Time elapsed (h:m:s.ms) {}'.format(time_elapsed)
        params["time_elpased"] = str(time_elapsed)
        
        utils.save_dict_to_json(d=mlog, json_path=os.path.join(opts.default_prefix, opts.current_time, 'mlog.json'))
        utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))
 
        # Transfer results by G-mail
        MailSend(subject = "Short report-%s" % "CPN Knowledge distillation", 
                    msg = mlog,
                    login_dir = opts.login_dir,
                    ID = 'singkuserver',
                    to_addr = ['sdimivy014@korea.ac.kr']).send()

    except KeyboardInterrupt:
        print("Stop !!!")
        os.rename(os.path.join(opts.default_prefix, opts.current_time), os.path.join(opts.default_prefix, opts.current_time + '_aborted'))

    except Exception as e:
        print("Error", e)
        print(traceback.format_exc())
        os.rename(os.path.join(opts.default_prefix, opts.current_time), os.path.join(opts.default_prefix, opts.current_time + '_error'))

    total_time = datetime.now() - total_time

    print('Time elapsed (h:m:s.ms) {}'.format(total_time))