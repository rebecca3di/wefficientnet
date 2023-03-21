from track import *
from translators import combine_translator as translator

import pickle as pk
import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from itertools import cycle

device_ = torch.device('cuda:0')

translator_ = translator().to(device_)
optimizer_ = torch.optim.Adam(translator_.parameters(), lr=0.001)
resize_ = torch.nn.AdaptiveAvgPool2d((368, 368))

from tensorboardX import SummaryWriter

class Recorder(object): # save the object value
    def __init__(self):
        self.last=0
        self.values=[]
        self.nums=[]
    def update(self,val,n=1):
        self.last=val
        self.values.append(val)
        self.nums.append(n)
    def avg(self):
        sum=np.sum(np.asarray(self.values)*np.asarray(self.nums))
        count=np.sum(np.asarray(self.nums))
        return sum/count

class jpg_n_csi(Dataset):
    def __init__(self, pk_path, mode='amp', posix='.pk1'):
        self.df = pk.load(open(pk_path, 'rb'))
        self.len = len(self.df)

    @staticmethod
    def iq_gate(x):
        real = np.real(x)
        imag = np.imag(x)
        return [torch.tensor(real).float(), torch.tensor(imag).float()]

    def __getitem__(self, idx):
        d = dict(self.df.iloc[idx])
        return d['jpg']

    def __len__(self):
        return self.len

def loss_mse(x, x_):
    return sum([F.mse_loss(torch.tensor(y_[1:]), torch.tensor(y[1:])) for (y_, y) in zip(x_[0], x[0])])

def loss_kld(x, x_, eps=1e-8):
    return sum([F.kl_div((y_ + eps).log(), y) for (y_, y) in zip(x, x_)])


def infer(batch, model, lite, framework):
    """
    Perform inference on supplied image batch.
    
    Args:
        batch: ndarray
            Stack of preprocessed images
        model: deep learning model
            Initialized EfficientPose model to utilize (RT, I, II, III, IV, RT_Lite, I_Lite or II_Lite)
        lite: boolean
            Defines if EfficientPose Lite model is used
        framework: string
            Deep learning framework to use (Keras, TensorFlow, TensorFlow Lite or PyTorch)
        
    Returns:
        EfficientPose model outputs for the supplied batch.
    """
    
    # Keras
    if framework in ['keras', 'k']:
        if lite:
            batch_outputs = model.predict(batch)
        else:
            batch_outputs = model.predict(batch)[-1]
    
    # TensorFlow
    elif framework in ['tensorflow', 'tf']:
        output_tensor = model.graph.get_tensor_by_name('upscaled_confs/BiasAdd:0')
        if lite:
            batch_outputs = model.run(output_tensor, {'input_1_0:0': batch})            
        else:
            batch_outputs = model.run(output_tensor, {'input_res1:0': batch})
    
    # TensorFlow Lite
    elif framework in ['tensorflowlite', 'tflite']:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], batch)
        model.invoke()
        batch_outputs = model.get_tensor(output_details[-1]['index'])
    
    # PyTorch
    elif framework in ['pytorch', 'torch']:
        from torch import from_numpy, autograd
        batch = np.rollaxis(batch, 3, 1)
        batch = from_numpy(batch)
        batch = autograd.Variable(batch, requires_grad=False).float()
        batch_outputs = model(batch)
        batch_outputs = batch_outputs.detach().numpy()
        batch_outputs = np.rollaxis(batch_outputs, 1, 4)

    # PyTorch_transparent
    elif framework in ['pytorch_transparent']:
        from torch import from_numpy, autograd
        if not isinstance(batch, torch.Tensor):
            batch = np.rollaxis(batch, 3, 1)
            batch = from_numpy(batch)
            batch = autograd.Variable(batch).float()
        # else:
        #     batch = batch.permute([0, 3, 1, 2])
        if batch.shape[1] != 3:
            batch = batch.permute([0, 3, 1, 2])
        batch_outputs = model(batch)
        batch_outputs = [_.permute([0, 2, 3, 1]) for _ in batch_outputs]
        
    return batch_outputs


def analyze_image_transparent(file_path, model, framework, resolution, lite, posix='.pk1', translator=translator_, train=True, optimizer=optimizer_):
    # Load image
    from PIL import Image
    start_time = time.time()

    image = np.array(Image.open(file_path))
    image_height, image_width = image.shape[:2]

    csi = pk.load(open(file_path[:-4] + posix, 'rb'))

    batch = np.expand_dims(image, axis=0)
    _batch = torch.tensor(np.expand_dims(csi[0], axis=0)).float().to(device_), torch.tensor(np.expand_dims(csi[1], axis=0)).float().to(device_)

    # Preprocess batch
    batch = torch.tensor(helpers.preprocess(batch, resolution, lite)).to(device_)
    batch_ = resize_(translator(_batch))
    
    # Perform inference
    batch_outputs = infer(batch, model, lite, framework)
    batch_outputs_ = infer(batch_, model, lite, framework)

    # Extract coordinates
    coordinates = [helpers.extract_coordinates(batch_outputs[0][0,...].detach().cpu().numpy(), image_height, image_width)]
    coordinates_ = [helpers.extract_coordinates(batch_outputs_[0][0,...].detach().cpu().numpy(), image_height, image_width)]

    if train:
        optimizer.zero_grad()
        loss = loss_kld(batch_outputs, batch_outputs_) + loss_mse(coordinates, coordinates_)
        loss.backward()
        optimizer.step()
    
    # Print processing time
    print('\n##########################################################################################################')
    print('Image processed in {0} seconds'.format('%.3f' % (time.time() - start_time)))
    print('##########################################################################################################\n')
    
    return coordinates_


def main(file_path, model_name, framework_name, visualize, store):
    
    # LIVE ANALYSIS FROM CAMERA
    if file_path is None:
        print('Not implemented.')

    # VIDEO ANALYSIS
    elif 'Video' in [track.track_type for track in MediaInfo.parse(file_path).tracks]:
        print('Not implemented.')
        
    # IMAGE ANALYSIS
    elif 'Image' in [track.track_type for track in MediaInfo.parse(file_path).tracks]:
        train_ds = jpg_n_csi('/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse/pathlist_train.pk')
        valid_ds = jpg_n_csi('/home/lscsc/caizhijie/ref-rep/pytorch-openpose/dataparse/pathlist_valid.pk')
        train_dl = DataLoader(train_ds, 1, True)
        valid_dl = DataLoader(valid_ds, 1, False)
        for j in range(20):
            valid_loss_recorder = Recorder()

            for i in tqdm.trange(1000):
                idx, file_path = next(enumerate(cycle(train_dl)))
                perform_tracking(False, normpath(file_path[0]), model_name, framework_name, visualize, store)
            for i in tqdm.trange(100):
                idx, file_path = next(enumerate(cycle(valid_dl)))
                perform_tracking(False, normpath(file_path[0]), model_name, framework_name, visualize, store, train=False)

                # writer.add_images('valid_image', img_batch / 255, epoch, dataformats='NHWC')
            
            valid_loss_epoch = valid_loss_recorder.avg()
            writer.add_scalars(
                'loss', {'train_loss_epoch': valid_loss_epoch, 'valid_loss_epoch': valid_loss_epoch}, global_step=j)
      
    else:
        print('\n##########################################################################################################')
        print('Ensure supplied file "{0}" is a video or image'.format(file_path))
        print('##########################################################################################################\n')
    

if __name__ == '__main__':

    task_save_root = '0321testrun'
    writer = SummaryWriter('tensorboard/'+task_save_root)

    # Fetch arguments
    args = sys.argv[1:]

    # Define options
    short_options = 'p:m:f:vs'
    long_options = ['path=', 'model=', 'framework=', 'visualize', 'store']
    try:
        arguments, values = getopt(args, short_options, long_options)
    except:
        print('\n##########################################################################################################')
        print(str(err))
        print('##########################################################################################################\n')
        sys.exit(2)

    # Define default choices
    file_path = None
    model_name = 'I_Lite'
    framework_name = 'TFLite'
    visualize = False
    store = False

    # Set custom choices
    for current_argument, current_value in arguments:
        if current_argument in ('-p', '--path'):
            file_path = current_value if len(current_value) > 0 else None
        elif current_argument in ('-m', '--model'):
            model_name = current_value
        elif current_argument in ('-f', '--framework'):
            framework_name = current_value
        elif current_argument in ('-v', '--visualize'):
            visualize = True
        elif current_argument in ('-s', '--store'):
            store = True
    print('\n##########################################################################################################')
    print('The program will attempt to analyze {0} using the "{1}" framework with model "{2}", and the user did{3} like to store the predictions and wanted{4} to visualize the result.'.format('"' + file_path + '"' if file_path is not None else 'the camera', framework_name, model_name, '' if store else ' not', '' if visualize or file_path is None else ' not'))
    print('##########################################################################################################\n')
        
    main(file_path=file_path, model_name=model_name, framework_name=framework_name, visualize=visualize, store=store)