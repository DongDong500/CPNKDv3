import datasets as dt
from utils import ext_transforms as et


def _get_dataset(opts, kftimes):
    
    mean = [0.485, 0.456, 0.406] if opts.is_rgb else [0.485]
    std = [0.229, 0.224, 0.225] if opts.is_rgb else [0.229]

    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtRandomVerticalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std)
        ])
    val_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.val_crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        ])
    test_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.val_crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        ])

    train_dst = dt.getdata.__dict__[opts.dataset](root=opts.data_root, datatype=opts.dataset, is_rgb=opts.is_rgb,
                                                    image_set='train', transform=train_transform, kfold=opts.k_fold, kftimes=kftimes)
    val_dst = dt.getdata.__dict__[opts.dataset](root=opts.data_root, datatype=opts.dataset, is_rgb=opts.is_rgb,
                                                image_set='val', transform=val_transform, kfold=opts.k_fold, kftimes=kftimes)
    test_dst = dt.getdata.__dict__[opts.dataset](root=opts.data_root, datatype=opts.dataset, is_rgb=opts.is_rgb,
                                                image_set='test', transform=val_transform, kfold=opts.k_fold, kftimes=kftimes)

    print("Dataset: %s\n\tTrain\t%d\n\tVal\t%d\n\tTest\t%d" % 
                            (opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_dst, val_dst, test_dst
