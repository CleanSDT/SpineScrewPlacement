from copy import deepcopy
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from utils import pedical_utils as utils3
import torch

import utils

class SpineEnv(gym.Env):
    def __init__(self, spineData, degree_threshold, **opts):
        """
        Args:
            spineData:
                - mask_coards: Coordinate of every pixel. shape:(3,m,n,t)
                - mask_array: Segmentation results of spine, 1-in 0-out. shape:(slice, height, width) （zyx）
                - pedicle_points: center points in pedicle (x,y,z). shape:(2,3)
        """
        self.reset_opt = opts['reset']
        self.step_opt = opts['step']

        self.state_shape = self.reset_opt['state_shape']
        self.discrete_action = self.step_opt['discrete_action'] #离散动作
        self.mask_array= spineData['mask_array']
        self.mask_coards = spineData['mask_coards']
        self.pedicle_points = spineData['pedicle_points']
        self.centerPointL = self.pedicle_points[0]
        self.centerPointR = self.pedicle_points[1]
        self.action_num = 2
        self.rotate_mag = self.step_opt['rotate_mag'] 
        self.reward_weight = [0.1, 0.1] 
        self.degree_threshold = degree_threshold

        self.min_action = -1.0 
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(self.action_num,))
        # self.trans_mag = np.array(self.step_opt.trans_mag)
        self.rotate_mag = np.array(self.rotate_mag) 
        self.weight = np.array(self.reward_weight)
        self.degree_threshold = np.array(self.degree_threshold)
        self.radiu_thres = None #[self.step_opt.radiu_thres[0] + spineData.cpoint_l[0], self.step_opt.radiu_thres[1] +spineData.cpoint_l[0]]
        self.line_thres =  None #self.step_opt.line_thres
        self.done_radius = 
        dist = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointR)
        self.cp_threshold = (self.centerPointR, np.min(dist)-1) # The position of the allowed points, represented as (center of sphere, radius)
        self.seed()
        self.state_matrix = None
        self.steps_before_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def computeInitDegree(self, random_reset = False): 
        if random_reset:
            horizon_left = self.np_random.uniform(low = self.degree_threshold[0], high = self.degree_threshold[1], size = [1,])[0]
            sagtal_left = self.np_random.uniform(low = self.degree_threshold[2], high = self.degree_threshold[3], size = [1,])[0]
            horizon_right = self.np_random.uniform(low = self.degree_threshold[0], high = self.degree_threshold[1], size = [1,])[0]
            sagtal_right = self.np_random.uniform(low = self.degree_threshold[2], high = self.degree_threshold[3], size = [1,])[0]
            return np.array([horizon_left,sagtal_left]), np.array([horizon_right,sagtal_right])
        else: return np.array([0.,0.]), np.array([0.,0.])

    def computeInitCPoint(self): 
        return self.centerPointL, self.centerPointR

    def normalize(self, array):
        mu = np.mean(array)
        std = np.std(array)
        return (array-mu) / std
    
    def reset(self, random_reset=False):
        self.steps_before_done = None 
        init_degree_L, init_degree_R = self.computeInitDegree(random_reset)
        init_cpoint_L, init_cpoint_R = self.computeInitCPoint()
        dire_vector_L, dire_vector_R = utils3.coorLngLat2Space(init_degree_L, init_degree_R, default=True)
        self.dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, init_cpoint_L)
        self.dist_mat_point_R = utils3.spine2point(self.mask_coards, self.mask_array, init_cpoint_R)
        self.dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, init_cpoint_L, dire_vector_L)
        self.dist_mat_line_R = utils3.spine2line(self.mask_coards, self.mask_array, init_cpoint_R, dire_vector_R)
        self.pre_max_radius_L, self.pre_line_len_L, self.endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, init_cpoint_L, dire_vector_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point_L, line_dist=self.dist_mat_line_L)
        self.pre_max_radius_R, self.pre_line_len_R, self.endpoints_R = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, init_cpoint_R, dire_vector_R, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point_R, line_dist=self.dist_mat_line_R)

        state_3D = self.draw_state(self.endpoints_L, self.endpoints_R)
        state_list = [self.pre_max_radius_L, self.pre_max_radius_R, self.pre_line_len_L, self.pre_line_len_R]
        state_list.extend(init_degree_L)
        state_list.extend(init_degree_R)
        self.state_matrix = np.asarray(state_list, dtype = np.float32)
        state_ = self.state_matrix * 1.0
        return np.asarray(state_, dtype=np.float32), self.normalize(state_3D)

    def stepPhysics(self, state_matrix, delta_degree_L, delta_degree_R, delta_cpoint = None):
        radius_L, radius_R, length_L, length_R, degree_L, degree_R = state_matrix[0], state_matrix[1], state_matrix[2], state_matrix[3], state_matrix[4:6], state_matrix[6:8]
        result = {'d_L': degree_L + delta_degree_L,
                  'd_R': degree_R + delta_degree_R,
                  }
        return result['d_L'], result['d_R']

    def getReward(self, state):
        line_len_L = state[2]
        line_len_R = state[3]
        len_delta_L = line_len_L - self.pre_line_len_L
        self.pre_line_len_L = line_len_L
        len_delta_R = line_len_R - self.pre_line_len_R
        self.pre_line_len_R = line_len_R
        radius_delta_L = state[0] - self.pre_max_radius_L
        self.pre_max_radius_L = state[0]
        radius_delta_R = state[1] - self.pre_max_radius_R
        self.pre_max_radius_R = state[1]
        return len_delta_L, len_delta_R, radius_delta_L, radius_delta_R

    def step(self, action_L, action_R):
        if self.discrete_action:
            discrete_vec = np.array()
            discrete_vec = np.rad2deg(discrete_vec)
            rotate_deg_L = np.array([discrete_vec[action_L[0]], discrete_vec[action_L[1]]]) 
            rotate_deg_R = np.array([discrete_vec[action_R[0]], discrete_vec[action_R[1]]])
        else:
            # action_L = np.clip(action_L, -1.0, 1.0)
            # action_R = np.clip(action_R, -1.0, 1.0)
            rotate_deg_L = self.rotate_mag * action_L[0:2]
            rotate_deg_R = self.rotate_mag * action_R[0:2]

        this_degree_L, this_degree_R = self.stepPhysics(self.state_matrix, rotate_deg_L,rotate_deg_R, delta_cpoint = None)
        this_dirpoint_L, this_dirpoint_R = utils3.coorLngLat2Space(this_degree_L, this_degree_R, R=1., default = True)
        self.dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        self.dist_mat_point_R = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointR)
        self.dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L)
        self.dist_mat_line_R = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointR, this_dirpoint_R)
        max_radius_L, line_len_L, self.endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point_L, line_dist=self.dist_mat_line_L)
        max_radius_R, line_len_R, self.endpoints_R = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointR, this_dirpoint_R, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=self.dist_mat_point_R, line_dist=self.dist_mat_line_R)

        state_list = [max_radius_L, max_radius_R, line_len_L, line_len_R]
        state_list.extend(this_degree_L)
        state_list.extend(this_degree_R)

        state_3D = self.draw_state(self.endpoints_L, self.endpoints_R)
        self.state3D_array = deepcopy(state_3D)

        self.state_matrix = np.asarray(state_list, dtype=np.float32)
        if max_radius_L < 0.:
            line_len_L = 0.01
            self.state_matrix[2] = 0.01
        if max_radius_R < 0.: 
            line_len_R = 0.01
            self.state_matrix[3] = 0.01

        
        done_L = self.state_matrix[0] < self.done_radius \
            or not (self.degree_threshold[0]<= self.state_matrix[4] <= self.degree_threshold[1]) \
            or not (self.degree_threshold[2]<= self.state_matrix[5] <= self.degree_threshold[3]) \
            # or not utils3.pointInSphere(self.state_matrix[3:], self.cp_threshold)
        done_R = self.state_matrix[1] < self.done_radius \
            or not (self.degree_threshold[0]<= self.state_matrix[6] <= self.degree_threshold[1]) \
            or not (self.degree_threshold[2]<= self.state_matrix[7] <= self.degree_threshold[3]) \
            # or not utils3.pointInSphere(self.state_matrix[3:], self.cp_threshold)
        done_L, done_R = bool(done_L), bool(done_R)
        done = done_L or done_R
        pre_volume = (3.14*(self.pre_line_len_L*self.pre_max_radius_L*self.pre_max_radius_L)+3.14*(self.pre_line_len_R*self.pre_max_radius_R*self.pre_max_radius_R))/2
        len_delta_L, len_delta_R, radius_delta_L, radius_delta_R = self.getReward(self.state_matrix)
        now_volume = (3.14*(self.state_matrix[2]*self.state_matrix[0]*self.state_matrix[0])+3.14*(self.state_matrix[3]*self.state_matrix[1]*self.state_matrix[1]))/2
        delta_volume = (now_volume - pre_volume)
        reward = 
        state_ = self.state_matrix * 1.0
        return np.asarray(state_, dtype=np.float32), reward, done, \
            {'len_delta_L': len_delta_L, 'len_delta_R': len_delta_R, \
             'radius_delta_L': radius_delta_L, 'radius_delta_R': radius_delta_R, \
             'action_left':rotate_deg_L, 'action_right':rotate_deg_R, 'delta_volume':delta_volume}, self.normalize(state_3D)

    def render_(self, fig, info=None, is_vis=False, is_save_gif=False, img_save_path=None, **kwargs):
        # fig = plt.figure()
        if is_vis:
            plt.ion()
        visual_ = self.mask_array #+ np.where(self.dist_mat <= )
        x_visual = np.max(visual_[:, :, :], 0)
        z_visual = np.max(visual_[:, :, :], 2)

        plt.clf()
        ax2 = fig.add_subplot(221)
        ax2.imshow(np.transpose(x_visual, (1, 0)))
        ax2.scatter(self.endpoints_L['radiu_p'][1], self.endpoints_L['radiu_p'][2], c='r')
        ax2.scatter(self.endpoints_L['start_point'][1], self.endpoints_L['start_point'][2], c='g')
        ax2.scatter(self.endpoints_L['end_point'][1], self.endpoints_L['end_point'][2], c='g')
        ax2.scatter(self.endpoints_L['cpoint'][1], self.endpoints_L['cpoint'][2], c='g')

        ax2.scatter(self.endpoints_R['radiu_p'][1], self.endpoints_R['radiu_p'][2], c='white')
        ax2.scatter(self.endpoints_R['start_point'][1], self.endpoints_R['start_point'][2], c='g')
        ax2.scatter(self.endpoints_R['end_point'][1], self.endpoints_R['end_point'][2], c='g')
        ax2.scatter(self.endpoints_R['cpoint'][1], self.endpoints_R['cpoint'][2], c='g')
        ax2.set_xlabel('Y-axis')
        ax2.set_ylabel('Z-axis')
        ax2.invert_yaxis()
        
        ax3 = fig.add_subplot(222)
        ax3.imshow(np.transpose(z_visual, (1, 0)))
        ax3.scatter(self.endpoints_L['radiu_p'][0], self.endpoints_L['radiu_p'][1], c='r')
        ax3.scatter(self.endpoints_L['start_point'][0], self.endpoints_L['start_point'][1], c='g')
        ax3.scatter(self.endpoints_L['end_point'][0], self.endpoints_L['end_point'][1], c='g')
        ax3.scatter(self.endpoints_L['cpoint'][0], self.endpoints_L['cpoint'][1], c='g')

        ax3.scatter(self.endpoints_R['radiu_p'][0], self.endpoints_R['radiu_p'][1], c='white')
        ax3.scatter(self.endpoints_R['start_point'][0], self.endpoints_R['start_point'][1], c='g')
        ax3.scatter(self.endpoints_R['end_point'][0], self.endpoints_R['end_point'][1], c='g')
        ax3.scatter(self.endpoints_R['cpoint'][0], self.endpoints_R['cpoint'][1], c='g')
        ax3.set_xlabel('X-axis')
        ax3.set_ylabel('Y-axis')
        ax3.invert_yaxis()
        if info is not None:
            ax3.text(2, -60, '#Reward:' + '%.4f' % info['r'], color='red', fontsize=20)
            ax3.text(2, -90, '#TotalR:' + '%.4f' % info['reward'], color='red', fontsize=20)
            ax3.text(2, -120, '#action_L:' + '%s' % info['action_left'], color='red', fontsize=15)
            ax3.text(2, -150, '#action_R:' + '%s' % info['action_right'], color='red', fontsize=15)
            ax3.text(2, 110, '#frame:%.4d' % info['frame'], color='red', fontsize=20)
            ax2.text(2, -30, '#radius:%.2f, %.2f' % (self.state_matrix[0],self.state_matrix[1]), color='red', fontsize=15)
            ax2.text(2, -45, '#length:%.2f, %.2f' % (self.state_matrix[2],self.state_matrix[3]), color='red', fontsize=15)
            ax2.text(2, -60, '#volume_L:%.1f' % (3.14*self.state_matrix[2]*self.state_matrix[0]*self.state_matrix[0]), color='red', fontsize=15)
            ax2.text(2, -75, '#volume_R:%.1f' % (3.14*self.state_matrix[3]*self.state_matrix[1]*self.state_matrix[1]), color='red', fontsize=15)
            ax2.text(2, -85, '#angle_L:%.1f, %.1f' % (self.state_matrix[4], self.state_matrix[5]), color='red', fontsize=15)
            ax2.text(2, -95, '#angle_R:%.1f, %.1f' % (self.state_matrix[6], self.state_matrix[7]), color='red', fontsize=15)
        if is_save_gif:
            if info is not None:
                fig.savefig(img_save_path + '/Epoch%d_%d.jpg' % (info['epoch'], info['frame']))
            else: fig.savefig(img_save_path + '/Epoch_{}.jpg'.format('test'))

        if is_vis:
            plt.show()
            plt.pause(0.9)
            plt.ioff()
        return fig

    def draw_state(self, endpoints_L, endpoints_R):
        state_3D = deepcopy(self.mask_array)
        state_3D[endpoints_L['radiu_p']] = 2
        state_3D[endpoints_R['radiu_p']] = 2
        return np.array(state_3D, dtype=np.float32)

    def simulate_reward(self, radian_L, radian_R, base_state):
        this_degree_L, this_degree_R = np.rad2deg(radian_L), np.rad2deg(radian_R)
        this_dirpoint_L, this_dirpoint_R = utils3.coorLngLat2Space(this_degree_L, this_degree_R, R=1., default = True)
        dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        dist_mat_point_R = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointR)
        dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L)
        dist_mat_line_R = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointR, this_dirpoint_R)
        max_radius_L, line_len_L, endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=dist_mat_point_L, line_dist=dist_mat_line_L)
        max_radius_R, line_len_R, endpoints_R = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointR, this_dirpoint_R, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=dist_mat_point_R, line_dist=dist_mat_line_R)

        state_list = [max_radius_L, max_radius_R, line_len_L, line_len_R]
        state_list.extend(this_degree_L)
        state_list.extend(this_degree_R)
        state_matrix = np.asarray(state_list, dtype=np.float32)
        if max_radius_L < 0.: 
            line_len_L = 0.01
            state_matrix[2] = 0.01
        if max_radius_R < 0.: 
            line_len_R = 0.01
            state_matrix[3] = 0.01
        pre_max_radius_L, pre_max_radius_R, pre_line_len_L, pre_line_len_R, = base_state[0],base_state[1],base_state[2],base_state[3]
        pre_volume = (3.14*(pre_line_len_L*pre_max_radius_L*pre_max_radius_L)+3.14*(pre_line_len_R*pre_max_radius_R*pre_max_radius_R))/2  
        now_volume = (3.14*(state_matrix[2]*state_matrix[0]*state_matrix[0])+3.14*(state_matrix[3]*state_matrix[1]*state_matrix[1]))/2
        # reward = np.log(now_volume) - np.log(pre_volume) 
        reward = now_volume/10000.0 
        return reward

    def simulate_volume(self, radian_L, radian_R, draw_screw = False):
        this_degree_L, this_degree_R = np.rad2deg(radian_L), np.rad2deg(radian_R)
        this_dirpoint_L, this_dirpoint_R = utils3.coorLngLat2Space(this_degree_L, this_degree_R, R=1., default = True)
        dist_mat_point_L = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointL)
        dist_mat_point_R = utils3.spine2point(self.mask_coards, self.mask_array, self.centerPointR)
        dist_mat_line_L = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L)
        dist_mat_line_R = utils3.spine2line(self.mask_coards, self.mask_array, self.centerPointR, this_dirpoint_R)
        max_radius_L, line_len_L, endpoints_L = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointL, this_dirpoint_L, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=dist_mat_point_L, line_dist=dist_mat_line_L)
        max_radius_R, line_len_R, endpoints_R = utils3.getLenRadiu \
            (self.mask_coards, self.mask_array, self.centerPointR, this_dirpoint_R, R=1, line_thres=self.line_thres,
             radiu_thres=self.radiu_thres, point_dist=dist_mat_point_R, line_dist=dist_mat_line_R)

        state_list = [max_radius_L, max_radius_R, line_len_L, line_len_R]
        state_list.extend(this_degree_L)
        state_list.extend(this_degree_R)
        state_matrix = np.asarray(state_list, dtype=np.float32)
        if max_radius_L < 0.: 
            line_len_L = 0.01
            state_matrix[2] = 0.01
        if max_radius_R < 0.: 
            line_len_R = 0.01
            state_matrix[3] = 0.01
        now_volume = (3.14*(state_matrix[2]*state_matrix[0]*state_matrix[0])+3.14*(state_matrix[3]*state_matrix[1]*state_matrix[1]))/2
        state3D_array = self.draw_state(endpoints_L, endpoints_R)
        if draw_screw:
            return now_volume, state3D_array, state_matrix
        else: return now_volume
