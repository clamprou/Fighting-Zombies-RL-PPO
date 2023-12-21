import math
import gym
from gym import spaces
import numpy as np
from malmo_agent import Agent


class FightingZombiesDisc(gym.Env):

    def __init__(self, render_mode=None, agents=1):
        self.agent = Agent(agents)
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1,1,(4,), float)
        self.observation_space = spaces.Box(-1,1, (len(self.agent.state),), float)

    def reset(self, seed=None, options=None):
        if not self.agent.first_time:
            self.agent.update_per_episode()
        self.agent.start_episode()
        return np.asarray(self.agent.state)

    def step(self, action):
        self.agent.tick_reward = 0  # Restore reward per tick, since tick just started
        self.agent.play_action(action)
        self.agent.observe_env()
        return np.asarray(self.agent.state), self.agent.tick_reward, not(self.agent.is_episode_running()), self.agent.is_alive()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pass