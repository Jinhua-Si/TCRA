"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import ToTensor, ToPILImage, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of sampled examples")
parser.add_argument("--delta", type=float, default=0.5, help="The balanced coefficient")
parser.add_argument("--zeta", type=float, default=3.0, help="The upper bound of neighborhood")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

def GAM(images, gt, model, min, max):
    """
    The attack algorithm of our proposed CMI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation
    :param max: the max the clip operation
    :return: the adversarial images
    """
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    zeta = opt.zeta
    delta = opt.delta
    N = opt.N
    grad = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_1 = x + torch.rand_like(x).uniform_(-eps*zeta, eps*zeta)
            x_1 = V(x_1, requires_grad = True)
            output_v3 = model(x_1)
            loss = F.cross_entropy(output_v3, gt)
            g1 = torch.autograd.grad(loss, x_1,
                                        retain_graph=False, create_graph=False)[0]
            x_2 = x_1.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)

            nes_x = x_2.detach()
            nes_x = V(nes_x, requires_grad = True)
            output_v3 = model(nes_x)
            loss = F.cross_entropy(output_v3, gt)
            g2 = torch.autograd.grad(loss, nes_x,
                                        retain_graph=False, create_graph=False)[0]
            h_1 = g2 -g1
            x_3 = x_1.detach() + alpha * (-h_1)/torch.abs(h_1).mean([1, 2, 3], keepdim=True)
            nes_x_1 = x_3.detach()
            nes_x_1 = V(nes_x_1, requires_grad=True)
            output_v4 = model(nes_x_1)
            loss = F.cross_entropy(output_v4, gt)
            g3 = torch.autograd.grad(loss, nes_x_1,
                                     retain_graph=False, create_graph=False)[0]
            x_4 = x_3.detach() + alpha * (-g3)/torch.abs(g3).mean([1, 2, 3], keepdim=True)
            nes_x_2 = x_4.detach()
            nes_x_2 = V(nes_x_2, requires_grad=True)
            output_v5 = model(nes_x_2)
            loss = F.cross_entropy(output_v5, gt)
            g4 = torch.autograd.grad(loss, nes_x_2,
                                     retain_graph=False, create_graph=False)[0]
            h1 = (1-delta)*g2 + delta*g4
            h2 = (1-delta)*g1 + delta*g3
            # 分解 h2 为与 h1 平行和正交的两个分量
            h1_norm = torch.linalg.norm(h1.view(h1.size(0), -1), dim=1, keepdim=True).view(h1.size(0), 1, 1, 1)
            h1_unit = h1 / (h1_norm + 1e-8)  # 单位向量
            h2_parallel = torch.sum(h2 * h1_unit, dim=(1, 2, 3), keepdim=True) * h1_unit
            h2_orthogonal = h2 - h2_parallel
            h3 = h2_orthogonal
            avg_grad += (1-delta)*h1 + delta*h3
        noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = GAM(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, './incv3_gam_outputs/')


if __name__ == '__main__':
    main()