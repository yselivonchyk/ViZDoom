#!/usr/bin/python

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function

from time import sleep

import cv2
import numpy as np
from action import *
from vizdoom import *

import collector as cl
from action import action as ac
import action.Dispatcher as dp
import action.ActionDispatcher as radp

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("../../examples/config/basic.cfg")
#game.load_config("../../examples/config/deadly_corridor.cfg")
game.load_config("../../examples/config/deathmatch.cfg")
#game.load_config("../../examples/config/defend_the_center.cfg")
#game.load_config("../../examples/config/defend_the_line.cfg")
#game.load_config("../../examples/config/health_gathering.cfg")
#game.load_config("../../examples/config/my_way_home.cfg")
#game.load_config("../../examples/config/predict_position.cfg")
#game.load_config("../../examples/config/take_cover.cfg")

# Sets other rendering options
game.set_render_hud(False)
# game.set_render_crosshair(False)
game.set_render_weapon(False)
# game.set_render_decals(False)
# game.set_render_particles(False)
# Enables freelook in engine
# game.add_game_args("+freelook 1")

# This is most fun. It looks best if you inverse colors.
game.set_screen_format(ScreenFormat.DEPTH_BUFFER8)  # depth
game.set_screen_format(ScreenFormat.RGB24)          # color
# # game.set_screen_format(ScreenFormat.RGBA32)
game.set_screen_format(ScreenFormat.CRCGCBDB)

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables spectator mode, so you can play. Sounds strange but it is agent who is supposed to watch not you.
game.set_window_visible(True)

spectator = False
game.set_mode(Mode.PLAYER)
if spectator:
    game.set_mode(Mode.SPECTATOR)

game.init()
# sleep time in ms
sleep_time = 40


episodes = 10
cl.init(record=True)
distance = 0


def printCross(depth):
    step = 5
    y = len(depth) / 2 + 2
    x = len(depth[0]) / 2 + 2
    depth[y, :] = 0
    depth[:, x] = 0
    depth[y, ::step] = 200
    depth[::step, x] = 200

sum = 0
dsp = dp.Dispatcher()
# dsp = radp.ActionDispatcher()

for i in range(episodes):
    print("Episode #" +str(i+1))

    game.new_episode()
    while not game.is_episode_finished():

        s = game.get_state()
        img_buffer = s.image_buffer
        misc = s.game_variables

        a = dsp.action()
        if spectator:
            game.advance_action()
        else:
            game.make_action(a)

        a = game.get_last_action()
        r = game.get_last_reward()

        # Gray8 shape is not cv2 compliant
        if game.get_screen_format() in [ScreenFormat.GRAY8, ScreenFormat.DEPTH_BUFFER8]:
            depth = img_buffer.reshape(img_buffer.shape[1], img_buffer.shape[2], 1)
            distance = depth[depth.shape[0] / 2, depth.shape[1] / 2]
        if game.get_screen_format() in [ScreenFormat.CRCGCB]:
            img_buffer = np.swapaxes(img_buffer, 1, 2)
            img_buffer = np.swapaxes(img_buffer, 0, 2)
            depth = img_buffer[:, :, ::-1]
        if game.get_screen_format() in [ScreenFormat.CRCGCBDB]:
            img_buffer = np.swapaxes(img_buffer, 1, 2)
            img_buffer = np.swapaxes(img_buffer, 0, 2)
            img_buffer = img_buffer[:, :, ::-1]
            depth = img_buffer[:, :, 0]
            img =  img_buffer[:, :, 1:] -0
        dsp.handle(depth, a)

        # Display the image here!
        depth *= 10 % 255 # make it brighter
        # printCross(depth) # print a cross
        cv2.imshow('Doom Buffer', img)
        cl.prntscr(depth, img)
        cv2.waitKey(sleep_time)


    print("episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
    sleep(1.0)

game.close()
