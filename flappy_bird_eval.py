#! pip install pygame

#from google.colab import drive
#drive.mount('/content/drive')

import pygame
from pygame.locals import *
from itertools import cycle
import random
import numpy as np
import pandas as pd
import cv2
import sys
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run Headless Pygame environment

#! mkdir /content/assets
#! mkdir /content/assets/sprites
#! mv *.png /content/assets/sprites

"""## Load Game Resources"""

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def load(BASE_PATH = '/content/'):#'/content/drive/MyDrive/'):
    # path of player with different states
    PLAYER_PATH = (
            BASE_PATH + 'assets/sprites/redbird-upflap.png',
            BASE_PATH + 'assets/sprites/redbird-midflap.png',
            BASE_PATH + 'assets/sprites/redbird-downflap.png'
    )

    # path of background
    BACKGROUND_PATH = BASE_PATH + 'assets/sprites/background-black.png'

    # path of pipe
    PIPE_PATH = BASE_PATH + 'assets/sprites/pipe-green.png'

    IMAGES, HITMASKS = {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load(BASE_PATH + 'assets/sprites/0.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/1.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/2.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/3.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/4.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/5.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/6.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/7.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/8.png').convert_alpha(),
        pygame.image.load(BASE_PATH + 'assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load(BASE_PATH + 'assets/sprites/base.png').convert_alpha()

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )
    return IMAGES, HITMASKS

"""## Game Parameters Setting"""

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, HITMASKS = load()
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

    def frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            terminal = True
            #self.__init__()
            reward = -1

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

"""# DQN Model"""

import torch
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weights_init(layer):
  if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
    torch.nn.init.normal_(layer.weight, mean = 0., std = 0.01)
    layer.bias.data.fill_(0.01)

class DQN_net(torch.nn.Module):
  def __init__(self, in_channels = 4, out_actions = 2):
    super(DQN_net, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size = 8, stride = 4, padding = 2)
    self.maxpool1 = torch.nn.MaxPool2d(2, 2)
    self.conv2 = torch.nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
    self.conv3 = torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
    self.fc1 = torch.nn.Linear(6400, 512)
    self.fc2 = torch.nn.Linear(512, out_actions)

  def forward(self, x):
    x = F.relu(self.conv1(x))  # (10, 10, 32)
    x = F.relu(self.conv2(x)) # (5, 5, 64)
    x = F.relu(self.conv3(x)) # (5, 5, 64)
    x = x.reshape(-1, 6400)  # (1, 1600)
    x = F.relu(self.fc1(x)) # (1, 512)
    x = self.fc2(x) # (1, 2)
    return x

"""# Replay Memory"""

class ReplayMemory:
  def __init__(self, capacity):
    self.capacity = capacity
    self.container = []
  def store(self, transition):
    self.container.append(transition)
    if len(self.container) > self.capacity:
      del self.container[0]
  def sample(self, batch_size):
    return random.sample(self.container, batch_size)
  def __len__(self):
    return len(self.container)

"""# DQN Training Object"""

class DQN:
  STACK_FRAMES = 4
  def __init__(self, memory_capacity, batch_size, epsilon, explore, replace_period, alpha, gamma, num_frames, num_actions):
    # Hyper-parameters
    self.replace_period = replace_period
    self.replace_counter = 1
    self.epsilon = epsilon
    self.epsilon_step = (epsilon - 0.0001) / explore
    self.alpha = alpha
    self.gamma = gamma

    # NN, loss, optimizer
    self.policy_net = DQN_net(num_frames, num_actions).to(device)
    self.target_net = DQN_net(num_frames, num_actions).to(device)
    self.policy_net.apply(weights_init)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.loss_function = torch.nn.MSELoss().to(device)
    self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = self.alpha)
    # Replay Memory
    self.replay_memory = ReplayMemory(memory_capacity)
    self.batch_size = batch_size

  def train(self):
    # Sample transition
    batch = self.replay_memory.sample(self.batch_size)
    state, action, reward, state_, terminal = zip(*batch)
    state = torch.tensor(state, dtype = torch.float32, requires_grad = True, device = device).reshape(self.batch_size, STACK_FRAMES, 80, 80)
    action = torch.cat(action).to(device)
    reward = torch.tensor(reward, dtype = torch.float32, requires_grad = False, device = device).reshape(self.batch_size, 1)
    state_ = torch.tensor(state_, dtype = torch.float32, requires_grad = False, device = device).reshape(self.batch_size, STACK_FRAMES, 80, 80)
    # (R + gamma * Q_) - Q
    Q = self.policy_net(state).gather(dim = 1, index = action.view(-1, 1))
    Q_ = self.target_net(state_).max(dim = 1)[0].view(-1, 1)
    TD_target = torch.zeros(self.batch_size, 1).to(device)
    # G = reward + self.gamma * Q_
    for i in range(self.batch_size):
      if not terminal[i]:
        TD_target[i, 0] = reward[i, 0] + self.gamma * Q_[i, 0]
      else:
        TD_target[i, 0] = reward[i, 0]
    # TD_target[terminal == False, 0] = G[terminal == False, 0]
    # TD_target[terminal == True, 0] = reward[terminal == True, 0]
    # loss
    loss = self.loss_function(Q, TD_target)
    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if self.replace_counter % self.replace_period == 0:
      self.update_target_net()
      self.replace_counter = 1
    self.replace_counter += 1

  def choose_action(self, obs, is_train = True):
    if is_train:
      if random.random() > self.epsilon:
        return self.policy_net(obs).max(dim = 1)[1]
      else:
        return torch.tensor([random.randint(0, 1)], dtype = torch.int64, device = device)
    else:
      return self.policy_net(obs).max(dim = 1)[1]

  def memory_store(self, transition):
    self.replay_memory.store(transition)

  def update_epsilon(self):
    if self.epsilon > 0.0001:
      self.epsilon -= self.epsilon_step

  def update_target_net(self):
    print('Update Target Net')
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def load_model(self, PATH):
    checkpoint = torch.load(PATH)
    self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epsilon = checkpoint['epsilon']
    return checkpoint['episode'], checkpoint['iterations']

  def save_model(self, episode, iterations):
    torch.save({
        'episode': episode,
        'iterations': iterations,
        'policy_net_state_dict': self.policy_net.state_dict(),
        'target_net_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epsilon': self.epsilon
    }, './checkpoint' + str(episode) + '.tar')

"""# Write CSV"""

EP = 30
game = GameState()
CSV = False
VIDEO = True
dqn = DQN(memory_capacity = 50000,
            batch_size = 32,
            epsilon = 0.0001,
            explore = 100000,
            replace_period = 100,
            alpha = 1e-6,
            gamma = 0.99,
            num_frames = 4,
            num_actions = 2)

BASE_PATH = '/content/' #'/content/drive/MyDrive/'
a = os.listdir(BASE_PATH + 'ckpts')
a.sort(key = lambda x: int(x.split('checkpoint')[1].split('.')[0]))
avg_reward_list = []

for ckpt in a:  
  _, model_itr = dqn.load_model(BASE_PATH + 'ckpts/' + ckpt)
  avg_R = 0
  iterations = 1
  print('Model: {}'.format(ckpt))
  for ep in range(EP):
    game.__init__()
    R = 0
    if VIDEO: fourcc = cv2.VideoWriter_fourcc(*'XVID') # video
    if VIDEO: video_writer = cv2.VideoWriter(BASE_PATH + 'result'+str(ep)+'.avi', fourcc, 30, (288, 512)) # video

    obs, reward, terminal = game.frame_step(np.array([1, 0]))
    if VIDEO: frame = cv2.cvtColor(cv2.flip(cv2.rotate(obs, cv2.ROTATE_90_CLOCKWISE), 1), cv2.COLOR_RGB2BGR) # video
    if VIDEO: video_writer.write(frame) # video
    obs = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, obs = cv2.threshold(obs, 1, 255, cv2.THRESH_BINARY)
    obs = np.reshape(obs, (1, 80, 80))
    obs = np.concatenate([obs] * 4, axis = 0)
    while not terminal:
      # Choose actions
      if iterations % 1 == 0:
        obs_tmp = torch.tensor(obs, dtype = torch.float32, device = device).reshape(1, 4, 80, 80)
        action = dqn.choose_action(obs_tmp, False)
      else:
        action = torch.tensor(0, dtype = torch.int64, device = device)
      # Get next state
      if action.cpu().numpy()[0] == 0:
        act = np.array([1, 0])
      elif action.cpu().numpy()[0] == 1:
        act = np.array([0, 1])
      obs_, reward, terminal = game.frame_step(act)
      if VIDEO: frame = cv2.cvtColor(cv2.flip(cv2.rotate(obs_, cv2.ROTATE_90_CLOCKWISE), 1), cv2.COLOR_RGB2BGR) # video
      if VIDEO: video_writer.write(frame) # video
      obs_ = cv2.cvtColor(cv2.resize(obs_, (80, 80)), cv2.COLOR_BGR2GRAY)
      _, obs_ = cv2.threshold(obs_, 1, 255, cv2.THRESH_BINARY)
      obs_ = np.reshape(obs_, (1, 80, 80))
      obs_ = np.concatenate([obs_, obs[:3, ...]], axis = 0)
      # Update
      obs = obs_
      R += reward
    print('Episode: {}, Total Reward: {}'.format(ep+1, R))
    avg_R += R
    if VIDEO: video_writer.release()
  print('Agent: {}, Avg Return: {}'.format(ckpt, avg_R/EP))
  avg_reward_list.append([ckpt, model_itr, avg_R/EP])
  if CSV:  
    df = pd.DataFrame(avg_reward_list, columns = ['Filename', 'Iterations', 'Average Reward'])
    df.to_csv(BASE_PATH + 'avg_reward.csv', index = False)
