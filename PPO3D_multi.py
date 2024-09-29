import numpy as np
import multiprocessing
import sys
import os
import SimpleITK as sitk
sys.path.append(os.getcwd())
import torch
from torch.optim import Adam
import gym
import matplotlib.pyplot as plt
import time
import core_multi as core
from tqdm import tqdm
from multiprocessing import Manager,Process
from logx import EpochLogger
from env import SingleSpineEnv
from dataLoad.loadNii import get_spinedata
from utils.img_utils import images_to_video
from torch.utils.tensorboard import SummaryWriter

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32) #观测值
        self.act_L_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.act_R_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32) #总回报
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_a_Left_buf = np.zeros(size, dtype=np.float32) #左动作的对数概率
        self.logp_a_Right_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act_L, act_R, rew, val, logp_a_Left, logp_a_Right):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_L_buf[self.ptr] = act_L
        self.act_R_buf[self.ptr] = act_R
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_a_Left_buf[self.ptr] = logp_a_Left
        self.logp_a_Right_buf[self.ptr] = logp_a_Right
        self.ptr += 1 

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act_L=self.act_L_buf, act_R=self.act_R_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp_L=self.logp_a_Left_buf, logp_R=self.logp_a_Right_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def build_data_load(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=64, input_y=80, input_x=160):
    spine_data = get_spinedata(dataDir, pedicle_points,pedicle_points_in_zyx, input_z, input_y, input_x) #160，80，64 xyz
    return spine_data

def build_Env(spine_data, degree_threshold, cfg):
    env = SingleSpineEnv.SpineEnv(spine_data, degree_threshold, **cfg)
    return env

def evluateothers(args, env_fn, actor_critic=core.MyMLPActorCritic, ac_kwargs=dict()):
    cfg = {'deg_threshold':,
           'reset':{'rdrange':,
                    'state_shape': ,
           'step':{'rotate_mag':, 'discrete_action':args.discrete_action}
           }
    
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    index = 0
    max_rewards=[]
    for dataDir, pedicle_point in zip(dataDirs, pedicle_points):
        index += 1
        dataDir = os.path.join(os.getcwd(), dataDir)
        
        spine_data = build_data_load(dataDir, pedicle_point, pedicle_point_in_zyx, input_z=, input_y=, input_x=) 

        envs = build_Env(spine_data, cfg['deg_threshold'], cfg)  
       
        obs_dim = envs.state_shape
        act_dim = envs.action_num
        ac = actor_critic(obs_dim, act_dim, discrete=args.discrete_action, LTO = args.Leaning_to_Optimize, **ac_kwargs).to(device)

        Train_RESUME = os.path.join(args.snapshot_dir, '') 
        if Train_RESUME:
            ckpt = torch.load(Train_RESUME)
            # epoch = ckpt['epoch']
            ac.load_state_dict(ckpt)
        
        _, o_3D = envs.reset(random_reset = False)
        o_3D = torch.Tensor(o_3D).to(device)
        step = 0
        fig = plt.figure()
        rewards = 0
        max_reward = 0
        while step<args.steps:
            step += 1
            with torch.no_grad():
                action_left, action_right = ac.evaluate(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)
            # TRY NOT TO MODIFY: execute the game and log data.
            state_, reward, done, others, o_3D = envs.step(action_left.squeeze_(0).cpu().numpy() if len(action_left.shape) == 2 else action_left, \
                                                    action_right.squeeze_(0).cpu().numpy() if len(action_right.shape) == 2 else action_right)
            action_left = others['action_left']
            action_right = others['action_right']
            o_3D, next_done = torch.Tensor(o_3D).to(device), torch.Tensor([1.] if done else [0.]).to(device)
            if reward > max_reward:
                max_reward = reward
                max_screw = envs.state3D_array
            rewards = rewards + reward
            info = {'reward': rewards, 'r': reward, 'len_delta_L': others['len_delta_L'], 'radiu_delta_L': others['radius_delta_L'],
                'len_delta_R': others['len_delta_R'], 'radiu_delta_R': others['radius_delta_R'],
                'epoch': 0, 'frame': step, 
                'action_left':'{:.3f}, {:.3f}'.format(action_left[0], action_left[1]),
                'action_right':'{:.3f}, {:.3f}'.format(action_right[0], action_right[1])}

            fig = envs.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path=args.imgs_dir)
            if done:
                break
        max_rewards.append(max_reward)
        state3D_itk = sitk.GetImageFromArray(np.transpose(max_screw, (2, 1, 0)))
        sitk.WriteImage(state3D_itk, os.path.join(args.imgs_dir ,os.path.basename(dataDir)))
        images_to_video(args.imgs_dir, '*.jpg', isDelete=True, savename = os.path.basename(dataDir))
    print(max_rewards)


def evaluate(args, env, agent, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, o_3D = env.reset(random_reset = False)
    o_3D = torch.Tensor(o_3D).to(device)
    step = 0
    fig = plt.figure()
    rewards = 0
    obs_dim = env.state_shape
    act_dim = env.action_num
    max_reward = 0
    while step < args.steps:
        step += 1
        with torch.no_grad():
            action_left, action_right = agent.evaluate(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)
        
        state_, reward, done, others, o_3D = env.step(action_left.squeeze_(0).cpu().numpy() if len(action_left.shape) == 2 else action_left, \
                                                action_right.squeeze_(0).cpu().numpy() if len(action_right.shape) == 2 else action_right)
        action_left = others['action_left']
        action_right = others['action_right']
        o_3D, next_done = torch.Tensor(o_3D).to(device), torch.Tensor([1.] if done else [0.]).to(device)
        if reward > max_reward:
            max_reward = reward
            max_screw = env.state3D_array
        rewards = rewards + reward
        info = {'reward': rewards, 'r': reward, 'len_delta_L': others['len_delta_L'], 'radiu_delta_L': others['radius_delta_L'],
                'len_delta_R': others['len_delta_R'], 'radiu_delta_R': others['radius_delta_R'],
                'epoch': 0, 'frame': step, 
                'action_left':'{:.3f}, {:.3f}'.format(action_left[0], action_left[1]),
                'action_right':'{:.3f}, {:.3f}'.format(action_right[0], action_right[1])}

        fig = env.render_(fig, info, is_vis=False, is_save_gif=True, img_save_path=args.imgs_dir)
        if done:
            break
    state3D_itk = sitk.GetImageFromArray(np.transpose(max_screw, (2, 1, 0)))
    sitk.WriteImage(state3D_itk, os.path.join(args.imgs_dir ,os.path.basename('')))
    images_to_video(args.imgs_dir, '*.jpg', isDelete=True, savename = 'Update%d'%(epoch))

def ppo(args, env_fn, actor_critic=core.MyMLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=100, epochs=400, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4,
        vf_lr=1e-3, train_pi_iters=10, train_v_iters=10, lam=0.97, max_ep_len=100,
        target_kl=0.05, logger_kwargs=dict(), save_freq=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.imgs_dir):
        os.makedirs(args.imgs_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    run_name = f"{'FineScrewPlacement'}__{seed}__{time.strftime('%Y-%m-%d %H-%M',time.localtime(time.time()))}"

    writer = SummaryWriter(rf"D:\pythonProject\FineScrewPlacement\spinningup\runs\{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
       
    cfg = {'deg_threshold':,
           'reset':{'rdrange':,
                    'state_shape':,
           'step':{'rotate_mag':, 'discrete_action':args.discrete_action}
           }

    print(cfg)
    print(args)
   
    torch.manual_seed(seed)
    np.random.seed(seed)



    # Instantiate environment
    spine_datas = []
    for dataDir, pedicle_point in zip(dataDirs, pedicle_points):
        dataDir = os.path.join(os.getcwd(), dataDir)
        pedicle_point_in_zyx = True 
        spine_datas.append(build_data_load(dataDir, pedicle_point, pedicle_point_in_zyx, input_z=, input_y=, input_x=))
        
   
    envs = []
    for spine_data in spine_datas:
        env = build_Env(spine_data, cfg['deg_threshold'], cfg)  
        envs.append(env)
    obs_dim = envs[0].state_shape
    act_dim = envs[0].action_num

    ac = actor_critic(obs_dim, act_dim, discrete=args.discrete_action, LTO = args.Leaning_to_Optimize, **ac_kwargs).to(device)

   
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch) #/ num_procs()) 100
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    def compute_loss_pi(data):
        obs, act_L, act_R, adv, logp_old_L, logp_old_R = (data['obs'] if args.Leaning_to_Optimize else data['obs'].unsqueeze(1)).to(device), data['act_L'].to(device), data['act_R'].to(device), data['adv'].to(device), data['logp_L'].to(device), data['logp_R'].to(device)

        pi_L, logp_L, pi_R, logp_R = ac.pi(obs, act_L, act_R)

        ratio_L = torch.exp(logp_L - logp_old_L)# 计算概率比率
        clip_adv_L = torch.clamp(ratio_L, 1-clip_ratio, 1+clip_ratio) * adv # 应用裁剪torch.clamp(input, min, max, out=None)
        loss_pi_L = -(torch.min(ratio_L * adv, clip_adv_L)).mean()

        ratio_R = torch.exp(logp_R - logp_old_R)
        clip_adv_R = torch.clamp(ratio_R, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi_R = -(torch.min(ratio_R * adv, clip_adv_R)).mean()

        approx_kl_L = (logp_old_L - logp_L).mean().item()
        ent_L = pi_L.entropy().mean().item()
        clipped_L = ratio_L.gt(1+clip_ratio) | ratio_L.lt(1-clip_ratio) # 返回bool。gt:greater than（大于）; lt:less than（小于）
        clipfrac_L = torch.as_tensor(clipped_L, dtype=torch.float32).mean().item() #  True和False分别转换为1.0和0.0，计算裁剪比率

        approx_kl_R = (logp_old_R - logp_R).mean().item()
        ent_R = pi_R.entropy().mean().item()
        clipped_R = ratio_R.gt(1+clip_ratio) | ratio_R.lt(1-clip_ratio)
        clipfrac_R = torch.as_tensor(clipped_R, dtype=torch.float32).mean().item()

        pi_info = dict(kl=(approx_kl_L+approx_kl_R)/2, ent=(ent_L+ent_R)/2, cf=(clipfrac_L+clipfrac_R)/2)
        return (loss_pi_L+loss_pi_R)/2, pi_info

    def compute_loss_v(data):
        obs, ret = data['obs'].unsqueeze(1).to(device), data['ret'].to(device)
        return ((ac.v(obs) - ret)**2).mean()


    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)


    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            loss_pi.backward()
            pi_optimizer.step()

        
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
    
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        return pi_l_old, v_l_old, kl, ent, cf,
        

    start_time = time.time()
    env_index = 0
    env = envs[env_index]
    _, o_3D = env.reset(random_reset = False) 
    o_3D = torch.Tensor(o_3D).to(device)
    ep_ret, ep_len = 0, 0 
    global_step = 0


    for epoch in tqdm(range(1, epochs+1)):
        frac = 1.0 - (epoch - 1.0) / epochs
        lrnow = frac * pi_lr
        pi_optimizer.param_groups[0]["lr"] = lrnow
        vf_optimizer.param_groups[0]["lr"] = lrnow
        for t in range(local_steps_per_epoch):
           
            a_Left, a_Right, v, logp_a_Left, logp_a_Right = ac.step(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)

            state_, r, d, info, next_o_3D = env.step(a_Left, a_Right)

            ep_ret += r
            ep_len += 1

            o_3D = o_3D.squeeze(0).squeeze(0).cpu().numpy() if len(o_3D.shape) == 5 else o_3D.cpu().numpy()
            buf.store(o_3D, a_Left, a_Right, r, v, logp_a_Left, logp_a_Right)
          
            o_3D = next_o_3D
            o_3D = torch.Tensor(o_3D).to(device)
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            global_step = (epoch-1)*local_steps_per_epoch+t
            writer.add_scalar("charts/step_return", r, global_step)

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended: #or timeout:
                    _, _, v, _, _ = ac.step(o_3D.unsqueeze_(0).unsqueeze_(0) if len(o_3D.shape) == 3 else o_3D)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    writer.add_scalar("charts/episodic_return", ep_ret, epoch)
                    writer.add_scalar("charts/episodic_length", ep_len, epoch)
               
                if epoch_ended:
                    if epoch % 100 == 0:
                        env_index += 1
                    env = envs[env_index%len(envs)]
                _, o_3D = env.reset(random_reset = False)
                o_3D = torch.Tensor(o_3D).to(device)
                ep_ret, ep_len = 0, 0 


        writer.add_scalar("charts/learning_rate", pi_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", LossV, global_step)
        writer.add_scalar("losses/policy_loss", LossPi, global_step)
        writer.add_scalar("losses/entropy", Entropy, global_step)
        # writer.add_scalar("losses/old_approx_kl", KL, global_step)
        writer.add_scalar("losses/approx_kl", KL, global_step)
        writer.add_scalar("losses/clipfrac", ClipFrac, global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if epoch % save_freq == 0:
            torch.save(ac.state_dict(), args.snapshot_dir+'/ppo_%d.pth' % (epoch))
            ac = ac.eval()
            evaluate(args, env, ac, epoch)
            ac = ac.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FineScrewPlacement')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--imgs_dir', type=str, default='')
    parser.add_argument('--snapshot_dir', type=str, default='')
    parser.add_argument('--KL', type=str, default='No KL')
    parser.add_argument('--clip', type=str, default='clip in env')
    parser.add_argument('--Leaning_to_Optimize', type=bool, default=False)
    parser.add_argument('--discrete_action', type=bool, default=True)
    parser.add_argument('--LTO_length', type=int, default=10)
    args = parser.parse_args()


    from run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(args, build_Env, actor_critic=core.MyMLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        save_freq=args.save_freq, logger_kwargs=logger_kwargs,max_ep_len=args.steps)
    
    # evluateothers(args, build_Env, actor_critic=core.MyMLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))
