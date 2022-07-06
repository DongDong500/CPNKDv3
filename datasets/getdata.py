from .cpn import CPN

from utils.ext_transforms import ExtCompose

def cpn(root:str = '/', datatype:str = 'CPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -Median nerve
        1044 + 261 = 1305 samples
    
    Args:
        root (str)  :   path to data parent directory (Ex: /data1/sdi/datasets) 
        datatype (str)  :   data folder name (default: CPN_all)
        dver (str)  : version of dataset (default: splits)
        image_set (str) :    train/val or test (default: train)
        transform (ExtCompose)  :   composition of transform class
        is_rgb (bool)   :  True for RGB, False for gray scale images
        tvs (int)   :  train/validate dataset ratio 
                2 block = 1 mini-block train set, 1 mini-block validate set
                5 block = 4 mini-block train set, 1 mini-block validate set
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")

    return CPN(root, datatype, dver, image_set, transform, is_rgb)
