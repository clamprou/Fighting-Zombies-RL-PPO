from __future__ import division
from __future__ import print_function

import json
import random
import time
import uuid
from builtins import range

import malmo.MalmoPython as MalmoPython
import matplotlib
import torch
from matplotlib import pyplot as plt

MS_PER_TICK = 10

NUM_AGENTS = 1
NUM_MOBS = 3
UNRESPONSIVE_AGENT = 10
UNRESPONSIVE_ZOMBIES = 100



class Agent:
    def __init__(self, agents=1):
        self.first_time = True
        self.episode_reward = 0  # Rewards per tick
        self.tick_reward = 0  # Rewards per episode
        self.total_reward = 0  # Total rewards, never restore to 0
        self.zombies_alive = NUM_MOBS
        self.zombie_los = 0
        self.zombie_los_in_range = 0
        self.survival_time_score = 0  # Lasted to the end of the mission without dying.
        self.zombie_kill_score = 0  # Good! Help rescue humanity from zombie-kind.
        self.malmo_agent = MalmoPython.AgentHost()
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10000 + agents-1))
        self.running = True
        self.episode = 0
        self.rewards_from_zombies = [0,0,0]
        self.wait_for_attack_or_shield = 1
        self.attack = True
        self.zombies_ids = []
        self.current_life = 20
        self.current_pos = [0, 0]
        self.yaw = 0
        self.zombies_pos = [[0, 0], [0, 0], [0, 0]]
        self.zombie_yaw = [0, 0, 0]
        self.unresponsive_count = UNRESPONSIVE_AGENT
        self.all_zombies_died = False
        self.actions = ["attack 1", "move 1", "move -1", "strafe 1", "strafe -1", "turn 1", "turn -1"]
        self.actions_view = ["Attack", "Go_Front", "Go_Back", "Go_Right", "Go_Left", "Look_Right", "Look_Left"]
        self.rewards = []
        self.kills = []
        self.prev_kills = 0
        self.player_life = []
        self.survival_time = []
        self.state = [self.zombie_los_in_range, self.zombie_los, self.current_pos[0]
            , self.current_pos[1], self.current_life, self.yaw, self.zombie_yaw[0], self.zombie_yaw[1], self.zombie_yaw[2]
            , self.zombies_pos[0][0], self.zombies_pos[0][1], self.zombies_pos[1][0], self.zombies_pos[1][1],
                      self.zombies_pos[2][0], self.zombies_pos[2][1]]

    def start_episode(self):
        self.mission = MalmoPython.MissionSpec(self.__get_xml("true" if self.first_time else "false"), True)
        self.first_time = False
        experimentID = str(uuid.uuid4())
        self.__safe_start_mission(self.mission, MalmoPython.MissionRecordSpec(), 0, experimentID)
        self.__safe_wait_for_start()
        self.malmo_agent.sendCommand("chat /kill @e[type=!player]")
        self.__spawn_zombies()
        self.__safe_wait_for_zombies()
        self.malmo_agent.sendCommand("chat /gamerule naturalRegeneration false")
        self.malmo_agent.sendCommand("chat /gamerule doMobLoot false")
        self.malmo_agent.sendCommand("chat /difficulty 1")
        self.malmo_agent.sendCommand("chat /effect Robot speed 99999 2")
        self.malmo_agent.sendCommand("chat /effect Robot instant_health 20")
            # self.malmo_agent.sendCommand("chat /effect Robot minecraft:absorption 999999 1")
        self.unresponsive_count = UNRESPONSIVE_AGENT
        self.all_zombies_died = False

    def is_episode_running(self):
        return self.unresponsive_count > 0 and not self.all_zombies_died

    def play_action(self, action):
        self.malmo_agent.sendCommand("use 1")
        if self.episode % self.wait_for_attack_or_shield == 0:
            if self.attack:
                self.malmo_agent.sendCommand("hotbar.1 1")
                self.wait_for_attack_or_shield = round(30 * abs(action[3])) + 1
                self.malmo_agent.sendCommand("attack 1")
            else:
                self.wait_for_attack_or_shield = round(30 * abs(action[3])) + 1
                self.malmo_agent.sendCommand("hotbar.2 1")
            if action[3] >= 0:
                self.attack = True
            else:
                self.attack = False
        self.malmo_agent.sendCommand("move " + str(action[2]))
        self.malmo_agent.sendCommand("strafe " + str(action[1]))
        self.malmo_agent.sendCommand("turn " + str(action[0]))
        time.sleep(0.012)
        self.malmo_agent.sendCommand("move 0")
        self.malmo_agent.sendCommand("strafe 0")
        self.malmo_agent.sendCommand("turn 0")
        self.episode += 1
        # time.sleep(MS_PER_TICK * 0.02)

    def observe_env(self):
        world_state = self.malmo_agent.getWorldState()

        # Agent gets rewards automatically by the platform defined in mission xml
        if world_state.number_of_rewards_since_last_state > 0:
            for rew in world_state.rewards:
                self.tick_reward += rew.getValue()

        # If Agent is steel alive we observe the changes on the environment and calculate our own rewards
        if world_state.number_of_observations_since_last_state > 0:
            self.tick_reward -= 0.05  # -0.1 per tick pass
            self.unresponsive_count = UNRESPONSIVE_AGENT
            ob = json.loads(world_state.observations[-1].text)

            if len(ob[u'entities']) >= 2:
                self.yaw = ob[u'entities'][0][u'yaw'] % 360
            if all(d.get('name') != 'Zombie' for d in ob["entities"]):
                self.all_zombies_died = True
            else:  # Update zombies position
                for d in ob["entities"]:
                    if d.get('name') == 'Zombie':
                        if d.get('id') in self.zombies_ids:
                            idx = self.zombies_ids.index(d.get('id'))
                            self.zombies_pos[idx] = [d.get('x'), d.get('z')]
                            self.zombie_yaw[idx] = d.get('yaw') % 360

            # Observe environment
            # cur_zombies_alive = list(d.get('name') == 'zombie' for d in ob["entities"]).count(True)
            # if cur_zombies_alive - self.zombies_alive != 0:
            #     self.tick_reward += 100 * abs(cur_zombies_alive - self.zombies_alive)
            # self.zombies_alive = cur_zombies_alive
            self.zombie_los_in_range = 0
            self.zombie_los = 0
            if u'LineOfSight' in ob:
                los = ob[u'LineOfSight']
                if los[u'hitType'] == "entity" and los[u'inRange'] and los[u'type'] == "Zombie":
                    self.zombie_los_in_range = 1
                elif los[u'hitType'] == "entity" and los[u'type'] == "Zombie":
                    self.zombie_los = 1
            # if ob[u'TimeAlive'] != 0:
            #     self.survival_time_score = ob[u'TimeAlive']
            if "Life" in ob:
                life = ob[u'Life']
                if life != self.current_life:
                    self.current_life = life
                    # Here Agent got hit and lost life, so we will punish him
                    self.tick_reward -= 7
                    # print("Life changed: -5 reward")
            if "MobsKilled" in ob:
                self.zombie_kill_score = ob[u'MobsKilled']
            if "XPos" in ob and "ZPos" in ob:
                self.current_pos = [ob[u'XPos'], ob[u'ZPos']]
            if "XPos" in ob or "ZPos" in ob or "YPos" in ob:
                if ob[u'XPos'] >= 11.3 or ob[u'XPos'] <= -10.3 or ob[u'ZPos'] >= 11.3 or ob[u'ZPos'] <= -10.3 or ob[u'YPos'] <= 200:
                    self.malmo_agent.sendCommand("chat /kill")
                    self.unresponsive_count = 0
                    self.tick_reward -= 100
            if self.current_life == 0:
                self.unresponsive_count = 0
        elif world_state.number_of_observations_since_last_state == 0:
            self.tick_reward -= 0.05
            self.unresponsive_count -= 1
        if self.unresponsive_count <= 0 and not self.all_zombies_died:  # Agent died but we are here one tick before episode ends, so we punish him
            self.tick_reward -= 100
        self.state = [self.zombie_los_in_range, self.zombie_los, self.current_pos[0]
            , self.current_pos[1], self.current_life, self.yaw, self.zombie_yaw[0], self.zombie_yaw[1], self.zombie_yaw[2]
            , self.zombies_pos[0][0], self.zombies_pos[0][1], self.zombies_pos[1][0], self.zombies_pos[1][1],
                      self.zombies_pos[2][0], self.zombies_pos[2][1]]
        self.total_reward += self.tick_reward  # Update total reward, never restore to 0
        self.episode_reward += self.tick_reward  # Update reward per episode, restore to 0 after each episode ends

    def update_per_episode(self):
        self.survival_time.append(self.survival_time_score)
        self.player_life.append(self.current_life)
        if self.zombie_kill_score != self.prev_kills:
            self.kills.append(self.zombie_kill_score - self.prev_kills)
            self.prev_kills = self.zombie_kill_score
        else:
            self.kills.append(0)
        self.rewards.append(self.episode_reward)

        self.malmo_agent.sendCommand("quit")
        hasEnded = False
        while not hasEnded:
            hasEnded = True  # assume all good
            time.sleep(0.1)
            world_state = self.malmo_agent.getWorldState()
            if world_state.is_mission_running:
                hasEnded = False  # all not good
        self.episode_reward = 0
        self.zombies_alive = NUM_MOBS
        self.reset_state()
        self.state = [self.zombie_los_in_range, self.zombie_los, self.current_pos[0]
            , self.current_pos[1], self.current_life, self.yaw, self.zombie_yaw[0], self.zombie_yaw[1], self.zombie_yaw[2]
            , self.zombies_pos[0][0], self.zombies_pos[0][1], self.zombies_pos[1][0], self.zombies_pos[1][1],
                      self.zombies_pos[2][0], self.zombies_pos[2][1]]

    def reset_state(self):
        self.zombie_los = 0
        self.zombie_los_in_range = 0
        self.current_life = 20
        self.current_pos = [0, 0]
        self.yaw = 0
        self.zombies_pos = [[0, 0], [0, 0], [0, 0]]
        self.zombie_yaw = [0, 0, 0]

    def print_finish_data(self):
        print()
        print("=========================================")
        print("Episode Reward:", self.episode_reward)
        # print("Total Reward:", self.total_reward)
        print("Player life: ", self.current_life)
        # print("Survival time score: ", self.survival_time_score)
        print("zombie kill score: ", self.zombie_kill_score)
        print("=========================================")
        print()
        time.sleep(MS_PER_TICK * 0.000001)

    def __safe_wait_for_zombies(self):
        unresponsive_count = UNRESPONSIVE_ZOMBIES
        self.zombies_ids = []
        while True:
            world_state = self.malmo_agent.getWorldState()
            if len(world_state.observations) != 0:
                ob = json.loads(world_state.observations[-1].text)
                if any(d.get('name') == 'Zombie' for d in ob["entities"]):
                    if len(ob["entities"]) == NUM_MOBS + NUM_AGENTS:
                        for d in ob["entities"]:
                            if d.get('name') == 'zombie':
                                self.zombies_ids.append(d.get('id'))
                        break
            if unresponsive_count <= 0:
                self.malmo_agent.sendCommand("quit")
                time.sleep(3)
                self.start_episode()
            unresponsive_count -= 1
            time.sleep(MS_PER_TICK * 0.001)

    def get_zombies_ids(self):
        world_state = self.malmo_agent.getWorldState()
        ob = json.loads(world_state.observations[-1].text)
        for d in ob["entities"]:
            if d.get('name') == 'Zombie':
                self.zombies_ids.append(d.get('id'))

    def __safe_start_mission(self, mission, mission_record, role, expId):
        used_attempts = 0
        max_attempts = 5
        while True:
            try:
                # Attempt start:
                self.malmo_agent.startMission(mission, self.client_pool, mission_record, role, expId)
                break
            except MalmoPython.MissionException as e:
                errorCode = e.details.errorCode
                if errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                    time.sleep(2)
                elif errorCode == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        time.sleep(2)
                elif errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        time.sleep(2)
                else:
                    exit(1)
            if used_attempts == max_attempts:
                exit(1)

    def __safe_wait_for_start(self):
        start_flag = False
        start_time = time.time()
        time_out = 120  # Allow a two minute timeout.
        while not start_flag and time.time() - start_time < time_out:
            state = self.malmo_agent.peekWorldState()
            start_flag = state.has_mission_begun
            errors = [e for e in state.errors]
            if len(errors) > 0:
                # for e in errors:
                #     # print(e.text)
                exit(1)
            time.sleep(0.1)
        if time.time() - start_time >= time_out:
            exit(1)


    def __spawn_zombies(self):
        for _ in range(NUM_MOBS):
            self.malmo_agent.sendCommand(
                "chat /summon Zombie "
                + str(random.randint(-9,9))
                + " 202 "
                + str(random.randint(-9,9))

            )


    def drawMobs(self):
        xml = ""
        for i in range(NUM_MOBS):
            x = str(random.randint(0,0))
            z = str(random.randint(0,0))
            xml += '<DrawEntity x="' + x + '" y="202" z="' + z + '" type="Zombie"/>'
        return xml

    def __get_xml(self, reset):
        xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <About>
            <Summary/>
          </About>
          <ModSettings>
            <MsPerTick>''' + str(MS_PER_TICK) + '''</MsPerTick>
          </ModSettings>
          <ServerSection>
            <ServerInitialConditions>
              <Time>
                <StartTime>13000</StartTime>
              </Time>
            </ServerInitialConditions>
            <ServerHandlers>
              <FlatWorldGenerator forceReset="''' + reset + '''" generatorString="" seed=""/>
              <DrawingDecorator>
                <DrawLine x1="-10" y1="200" z1="10" x2="-10" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-9" y1="200" z1="10" x2="-9" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-8" y1="200" z1="10" x2="-8" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-7" y1="200" z1="10" x2="-7" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-6" y1="200" z1="10" x2="-6" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-5" y1="200" z1="10" x2="-5" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-4" y1="200" z1="10" x2="-4" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-3" y1="200" z1="10" x2="-3" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-2" y1="200" z1="10" x2="-2" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="-1" y1="200" z1="10" x2="-1" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="0" y1="200" z1="10" x2="0" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="1" y1="200" z1="10" x2="1" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="2" y1="200" z1="10" x2="2" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="3" y1="200" z1="10" x2="3" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="4" y1="200" z1="10" x2="4" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="5" y1="200" z1="10" x2="5" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="6" y1="200" z1="10" x2="6" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="7" y1="200" z1="10" x2="7" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="8" y1="200" z1="10" x2="8" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="9" y1="200" z1="10" x2="9" y2="200" z2="-10" type="wool" colour="ORANGE"/>
                <DrawLine x1="10" y1="200" z1="10" x2="10" y2="200" z2="-10" type="wool" colour="ORANGE"/>
              </DrawingDecorator>
            </ServerHandlers>
          </ServerSection>
        '''
        for i in range(NUM_AGENTS):
            xml += '''<AgentSection mode="Adventure">
            <Name>Robot</Name>
            <AgentStart>
              <Placement x="''' + str(0) + '''" y="202" z="''' + str(0) + '''"/>
              <Inventory>
                <InventoryBlock quantity="1" slot="0" type="stone_sword" />
                <InventoryBlock quantity="1" slot="1" type="shield" />
                <InventoryBlock quantity="1" slot="39" type="iron_helmet" />
                <InventoryBlock quantity="1" slot="38" type="iron_chestplate" />
                <InventoryBlock quantity="1" slot="37" type="iron_leggings" />
                <InventoryBlock quantity="1" slot="36" type="iron_boots" />
              </Inventory>
            </AgentStart>
            <AgentHandlers>
              <ContinuousMovementCommands turnSpeedDegs="360"/>
              <InventoryCommands/>
              <ChatCommands/>
              <MissionQuitCommands/>
                <RewardForDamagingEntity>
                    <Mob reward="50" type="Zombie"/>
                </RewardForDamagingEntity>
              <ObservationFromNearbyEntities>
                <Range name="entities" xrange="300" yrange="300" zrange="300"/>
              </ObservationFromNearbyEntities>
              <ObservationFromRay/>
              <ObservationFromFullStats/>
            </AgentHandlers>
          </AgentSection>'''

        if NUM_AGENTS != 1:
            xml += '''<AgentSection mode="Creative">
                <Name>TheWatcher</Name>
                <AgentStart>
                  <Placement x="0.5" y="228" z="0.5" pitch="90"/>
                </AgentStart>
                <AgentHandlers>
                  <ContinuousMovementCommands turnSpeedDegs="360"/>
                  <MissionQuitCommands/>
                  <VideoProducer>
                    <Width>640</Width>
                    <Height>640</Height>
                  </VideoProducer>
                </AgentHandlers>
              </AgentSection>'''

        xml += '</Mission>'
        return xml

    def sleep(self):
        time.sleep(MS_PER_TICK * 0.000001)

    def is_alive(self):
        if self.all_zombies_died:
            return True
        elif self.unresponsive_count <= 0:
            return False
        else:
            return True


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_table(table , variable,show_result=False):
    is_ipython = 'inline' in matplotlib.get_backend()
    plt.figure(1)
    values = torch.tensor(table, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel(variable)
    plt.scatter([i for i in range(len(values.numpy()))], values.numpy())
    # Take 100 episode averages and plot them too
    if len(values) >= 100:
        means = values.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color='orange')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())