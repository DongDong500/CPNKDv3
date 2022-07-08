import os
import torch
import torch.nn as nn

from argparse import ArgumentParser

import network
import utils

def _load_model(opts: ArgumentParser = None, model_name: str = '', msg: str = '', 
                verbose: bool = False, pretrain: str = None, 
                output_stride: int = 8, sep_conv: bool = False):
    
    print("<load model>", msg) if verbose else 0

    try:    
        if model_name.startswith("deeplab"):
            model = network.model.__dict__[model_name](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes, output_stride=output_stride)
            if sep_conv and 'plus' in model_name:
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)
        else:
            model = network.model.__dict__[model_name](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes)
    except:
        raise Exception("<load model> Error occured while loading a model.")

    if pretrain is not None and os.path.isfile(pretrain):
        print("<load model> restored parameters from %s" % pretrain)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        checkpoint = torch.load(pretrain, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint  # free memory
        torch.cuda.empty_cache()
        
    return model