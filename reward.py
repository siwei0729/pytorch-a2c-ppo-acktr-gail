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
        # self.encoder_net.load_state_dict(torch.load(model_path))
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

    def reward(self, s, j):
        # gray = self.rgb2gray(s)
        # resized_img = cv2.resize(gray, (256, 256))

        resized_img = s.cpu().detach().numpy()
        resized_img = resized_img / 255.0

        # resized_img = resized_img[:, :, :, None]
        rgb_image = cv2.merge((resized_img, resized_img, resized_img))
        rgb_image = rgb_image.reshape((-1, 84, 84, 3))
        pixel_val = rgb_image[0, 10, 60, 0]

        if pixel_val == 0:
            goal_wall = True
        elif abs(pixel_val - 0.75) < 0.01:
            goal_wall = True
        else:
            goal_wall = False

        img_s = torch.tensor(rgb_image).to(self.device)
        img_s = img_s.double()
        img_s_z = self.encoder_net(img_s)

        rewards = self.calc_reward1(img_s_z, goal_wall)
        return rewards

    def calc_reward1(self, img_s_z, goal_wall):

        reward_list = []
        for frame in range(img_s_z.shape[0]):

            distance_list = []
            for i in range(self.img_g_z.shape[0]):
                distance = torch.clamp(self.distance(img_s_z[frame], self.img_g_z[i]), max=5).cpu().detach().numpy()
                distance = np.sum(distance)
                distance_list.append(distance)

            reward = 0
            if distance_list[0] < 2.5:
                reward += 50
            if distance_list[1] < 3:
                reward += 50
            # if distance_list[2] < 3:
            #     reward -= 2
            reward_list.append(reward)

        return reward_list

    def calc_reward2(self, img_s_z):

        reward_list = []
        for frame in range(img_s_z.shape[0]):
            reward_sum_per_frame = 0
            for i in range(self.img_g_z.shape[0]):
                distance = torch.clamp(self.distance(img_s_z[frame], self.img_g_z[i]), max=5).cpu().detach().numpy()
                reward = np.exp(-1 * distance / 10.0)
                reward = np.sum(reward)
                reward_sum_per_frame += reward
            reward_list.append(reward_sum_per_frame)

        return reward_list
