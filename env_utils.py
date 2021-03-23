#! /usr/bin/python3
import pygame
from pygame.locals import *
from itertools import cycle
import random
import numpy as np
import sys

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def load(BASE_PATH = 'assets/sprites/'):
    # path of player with different states
    PLAYER_PATH = (
            BASE_PATH + 'redbird-upflap.png',
            BASE_PATH + 'redbird-midflap.png',
            BASE_PATH + 'redbird-downflap.png'
    )

    # path of background
    BACKGROUND_PATH = BASE_PATH + 'background-black.png'

    # path of pipe
    PIPE_PATH = BASE_PATH + 'pipe-green.png'

    IMAGES, HITMASKS = {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load(BASE_PATH + '0.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '1.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '2.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '3.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '4.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '5.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '6.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '7.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '8.png').convert_alpha(),
        pygame.image.load(BASE_PATH + '9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load(BASE_PATH + 'base.png').convert_alpha()

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
