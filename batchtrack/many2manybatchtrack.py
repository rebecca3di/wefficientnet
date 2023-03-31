import numpy as np
import torch.nn as nn
import torch
import tqdm
import tensorboardX
import copy

from torch.utils.data import DataLoader
from itertools import cycle

import sys
# sys.path.insert(0, '/home/lscsc/caizhijie/ref-rep/EfficientPose/')
sys.path.insert(0, 'D:\caizhijie\codes\wefficientnet')

from track import get_model
from many2manybatching import csi_n_pic_dataset, collate_fn
from utils.helpers import preprocess, extract_coordinates
from batchutils import loss_kld, loss_mse, Recorder, annotate_image
from translators import seq_translator as translator

from utils import helpers

def main(gpu_id=0,
         framework='pytorch_transparent',
         model_variant='ii',
        #  pk_path='/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse_/pathpacks_',
         pk_path='D:\caizhijie\codes\wopen-pose\dataparse_\pathpacks_',
         batch_size=1, 
         n_epoch=10000,
         len_epoch_train=10,
         len_epoch_valid=10,
         lite=False,
         translator=translator,
         lr=1e-4,
         weight_decay=1e-4,
         logdir='0327-many2many',
         preview_gap=10,
         accum_step=8,
         ):
    
    device = torch.device('cuda:%d' % gpu_id)
    if framework not in ['pytorch_transparent']:
        print('framework not yet implemented.')
    elif model_variant not in ['ii']:
        print('model_variant not yet implemented.')
    else:
        model_variant = model_variant[13:] if len(model_variant) > 7 else model_variant 
        lite = True if model_variant.endswith('_lite') else False
        model, resolution = get_model(framework, model_variant)
        if not model:
            return True
        
    writer = tensorboardX.SummaryWriter('tensorboard/' + logdir)
        
    translator = translator.to(device)
    translator.train()
    model = model.to(device)
    model.eval()

    train_ds = csi_n_pic_dataset(pk_path + 'train.pk')
    valid_ds = csi_n_pic_dataset(pk_path + 'valid.pk')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(translator.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay,)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                       gamma=0.998,
                                                    #    last_epoch=5000,
                                                       )

    # resizer = nn.AdaptiveAvgPool2d((368, 368))
    for j in range(n_epoch):
        train_loss_epoch = Recorder()
        valid_loss_epoch = Recorder()
        train_metric_epoch = Recorder()
        valid_metric_epoch = Recorder()
        for i in tqdm.trange(len_epoch_train * accum_step):
            idx, (jpg, csi) = next(enumerate(cycle(train_loader)))
            _jpg = copy.deepcopy(jpg).reshape(-1, jpg.shape[-3], jpg.shape[-2], jpg.shape[-1])

            jpg = jpg.reshape((-1, jpg.shape[-3], jpg.shape[-2], jpg.shape[-1]))
            jpg = torch.tensor(preprocess(jpg, resolution, lite)).to(device).permute([0, 3, 1, 2])
            jpg_ = translator(torch.tensor(csi).to(device).float()).permute([0, 2, 1, 3, 4])
            jpg_ = jpg_.reshape((-1, jpg.shape[-3], jpg.shape[-2], jpg.shape[-1]))

            outputs = model(jpg)
            outputs_ = model(jpg_)

            picsize = 368

            coordinates = [extract_coordinates(outputs[0][_,...].permute([1, 2, 0]).detach().cpu().numpy(), picsize, picsize) for _ in range(batch_size * _jpg.shape[0])]
            coordinates_ = [extract_coordinates(outputs_[0][_,...].permute([1, 2, 0]).detach().cpu().numpy(), picsize, picsize) for _ in range(batch_size * _jpg.shape[0])]

            loss, metric = loss_mse(outputs, outputs_)# + loss_mse(coordinates, coordinates_)
            train_loss_epoch.update(loss.detach().cpu().numpy())
            train_metric_epoch.update(metric.detach().cpu().numpy())

            loss.backward()
            
            if (1+i) % accum_step == 0:
                optimizer.zero_grad()
                optimizer.step()
            

            # preview image
            if i % preview_gap == 0:
                preview_batch = list()
                jpg_numpy = np.uint8(_jpg).transpose([0, 3, 1, 2])
                for k in range(4):
                    preview_batch.append(annotate_image(jpg_numpy[k], coordinates[k]))
                    preview_batch.append(annotate_image(jpg_numpy[k], coordinates_[k]))
                preview_batch = np.stack(preview_batch, axis=0)
                writer.add_images('train_image', preview_batch, j, dataformats='NHWC')
        scheduler.step()

        for i in tqdm.trange(len_epoch_valid * accum_step):
            with torch.no_grad():
                idx, (jpg, csi) = next(enumerate(cycle(valid_loader)))
                _jpg = copy.deepcopy(jpg).reshape(-1, jpg.shape[-3], jpg.shape[-2], jpg.shape[-1])

                jpg = jpg.reshape((-1, jpg.shape[-3], jpg.shape[-2], jpg.shape[-1]))
                jpg = torch.tensor(preprocess(jpg, resolution, lite)).to(device).permute([0, 3, 1, 2])
                jpg_ = translator(torch.tensor(csi).to(device).float()).permute([0, 2, 1, 3, 4])
                jpg_ = jpg_.reshape((-1, jpg.shape[-3], jpg.shape[-2], jpg.shape[-1]))

                outputs = model(jpg)
                outputs_ = model(jpg_)

                picsize = 368

                coordinates = [extract_coordinates(outputs[0][_,...].permute([1, 2, 0]).detach().cpu().numpy(), picsize, picsize) for _ in range(batch_size * _jpg.shape[0])]
                coordinates_ = [extract_coordinates(outputs_[0][_,...].permute([1, 2, 0]).detach().cpu().numpy(), picsize, picsize) for _ in range(batch_size * _jpg.shape[0])]

                loss, metric = loss_mse(outputs, outputs_)# + loss_mse(coordinates, coordinates_)

            valid_loss_epoch.update(loss.detach().cpu().numpy())
            valid_metric_epoch.update(metric.detach().cpu().numpy())

            # preview image
            preview_batch = list()
            if i % preview_gap == 0:
                jpg_numpy = np.uint8(_jpg).transpose([0, 3, 1, 2])
                for k in range(4):
                    preview_batch.append(annotate_image(jpg_numpy[k], coordinates[k]))
                    preview_batch.append(annotate_image(jpg_numpy[k], coordinates_[k]))
                preview_batch = np.stack(preview_batch, axis=0)
                writer.add_images('valid_image', preview_batch, j, dataformats='NHWC')

        writer.add_scalars('loss', {'valid_loss_epoch': valid_loss_epoch.avg(), 'train_loss_epoch': train_loss_epoch.avg()}, global_step=j)
        writer.add_scalars('metric', {'valid_metric_epoch': valid_metric_epoch.avg(), 'train_metric_epoch': train_metric_epoch.avg()}, global_step=j)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], j)

        if j % 100 == 0:
            torch.save(translator, 'tensorboard/' + logdir + '/t_epoch%d.model' % j)
            if j == 0:
                torch.save(model, 'tensorboard/' + logdir + '/m_epoch%d.model' % j)



if __name__ == '__main__':
    translator_ = translator()
    main(translator=translator_)