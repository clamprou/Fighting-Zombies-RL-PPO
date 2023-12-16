from gym_env import FightingZombiesDisc
from malmo_agent import plot_table
from utils import evaluate_policy, str2bool
from datetime import datetime
from PPO import PPO_discrete
import gymnasium as gym
import os, shutil
import argparse
import torch

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=512, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e7/4, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5/4, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3/4, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    # Build Training Env and Evaluation Env
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CP-v1', 'LLd-v2']
    env = FightingZombiesDisc()
    eval_env = FightingZombiesDisc(agents=2)
    opt.state_dim = env.observation_space_n
    opt.action_dim = env.action_space.n
    opt.max_e_steps = 200

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Env:FightingZombiesDqnDisc  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
    print('\n')

    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    agent = PPO_discrete(**vars(opt))
    if opt.Loadmodel: agent.load(opt.ModelIdex)

    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, turns=1)
            print(f'Env:FightingZombiesDqnDisc, Episode Reward:{ep_r}')
    else:
        episode, traj_lenth, total_steps = 0, 0, 0
        while total_steps < opt.Max_train_steps:
            s = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & train'''
            while not done:
                '''Interact with Env'''
                a, logprob_a = agent.select_action(s, deterministic=False) # use stochastic when training
                s_next, r, done, dw = env.step(a) # dw: dead&win; tr: truncated
                #if r <=-100: r = -30  #good for LunarLander TODO check if its not needed
                '''Store the current transition'''
                agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
                s = s_next

                traj_lenth += 1
                total_steps += 1

                '''Update if its time'''
                if traj_lenth % opt.T_horizon == 0:
                    agent.train()
                    traj_lenth = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3) # evaluate the policy for 3 times, and get averaged result
                    if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:FightingZombiesDqnDisc seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(total_steps)

            if episode >= 100:
                values = torch.tensor(env.agent.rewards, dtype=torch.float)
                means = values.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                print("| Episode: ", episode, "| Episode reward: ",
                      env.agent.episode_reward, " |", " Average reward: ", means.numpy()[-1])
                if episode % 100 == 0:
                    plot_table(env.agent.rewards, "rewards")
            else:
                print("| Episode: ", episode, "| Episode reward: ", env.agent.episode_reward, "|")

            episode += 1

        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
