import torch.nn.functional as F
import numpy as np
import torch
import copy

from PIL import Image, ImageDraw

from utils import helpers


def loss_mse(x, x_):
    loss = 0.0
    for i in range(x.__len__()):
        metric = F.smooth_l1_loss(x[i], x_[i])
        loss += metric
    return loss, metric
    # return sum([F.mse_loss(torch.tensor(y_[1:]), torch.tensor(y[1:])) for (y_, y) in zip(x_[0], x[0])])


def loss_mse_1(x, x_, i=-1):
    loss = 0.0
    # for i in range(x.__len__()):
    loss += F.smooth_l1_loss(x[i], x_[i])
    return loss


def loss_kld(x, x_, eps=1e-8):
    x = [_.view(_.shape[0], _.shape[1], -1) for _ in x]
    x_ = [_.view(_.shape[0], _.shape[1], -1) for _ in x_]

    return sum([F.kl_div(F.log_softmax(y_ + eps, dim=-1), F.softmax(y, dim=-1), reduce='batchmean') for (y_, y) in zip(x, x_)])


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
    
def annotate_image(image, coordinates):
    
    # Annotate image
    image_width, image_height = image.shape[-2], image.shape[-1]
    image_side = image_width if image_width >= image_height else image_height

    image = Image.fromarray(copy.deepcopy(image).transpose([1, 2, 0]))

    image_draw = ImageDraw.Draw(image)
    image_coordinates = coordinates
    image = helpers.display_body_parts(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, marker_radius=int(image_side/150))
    image = helpers.display_segments(image, image_draw, image_coordinates, image_height=image_height, image_width=image_width, segment_width=int(image_side/100))
    
    # Save annotated image
    # image.save('k.png')

    return np.array(image)