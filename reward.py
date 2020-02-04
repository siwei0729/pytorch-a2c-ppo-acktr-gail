import os
import numpy as np
import cv2
import torch

from model import Net

folder_size = 0
model_path = "models/encoder_net_abstract.pth"
for subdir, dirs, files in os.walk('goal_abstract'):
    folder_size = len(dirs)
    break


class LFTReward:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        self.encoder_net = Net()
        self.encoder_net = self.encoder_net.to(self.device)
        self.encoder_net.double()
        self.encoder_net.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))
        self.encoder_net.eval()

        img_g = self.load_data()
        img_g = torch.tensor(img_g).to(self.device)
        self.img_g_z = self.encoder_net(img_g)

    @staticmethod
    def distance(x1, x2):
        diff = torch.abs(x1 - x2).contiguous()
        return torch.pow(diff, 2)

    @staticmethod
    def load_data():
        img_ds = []
        for i in range(1, folder_size + 1):
            img_d = cv2.imread(f'goal_abstract/{i}/bw_{i}.png')
            img_ds.append(img_d / 255.0)
        return img_ds

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def reward(self, s):
        gray = self.rgb2gray(s)
        resized_img = cv2.resize(gray, (256, 256))
        resized_img = resized_img[:, :, None]
        rgb_image = cv2.merge((resized_img, resized_img, resized_img))
        rgb_image = rgb_image[None, :, :, :]
        img_s = torch.tensor(rgb_image).to(self.device)
        img_s_z = self.encoder_net(img_s)

        reward = self.calc_reward1(img_s_z)
        return reward

    def calc_reward1(self, img_s_z):
        reward_sum = 0
        reward_list = []
        for i in range(self.img_g_z.shape[0]):
            reward = torch.clamp(self.distance(img_s_z, self.img_g_z[i]), max=5).cpu().detach().numpy()
            reward = np.sum(-i * reward)
            reward_list.append(reward)
            reward_sum += reward

        return reward_sum


def load_validation_data():
    img_os = []
    for i in range(1, 10):
        img_o = cv2.imread(f'validate_images/observation/{i}/{i}.png')
        img_os.append(img_o / 255.0)
    return img_os


def test():
    # img_os = load_validation_data()
    ltf_reward = LFTReward()
    # for s in img_os:
    #     reward = ltf_reward.reward(s)


test()
