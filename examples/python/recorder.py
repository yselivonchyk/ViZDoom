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
from action import DispatcherCircle, DispatcherLine, ActionDispatcher, DummyDispatcher

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("../../examples/config_all_actions/basic.cfg")
# game.load_config("../../examples/config_all_actions/deadly_corridor.cfg")
game.load_config("../../examples/config_all_actions/deathmatch.cfg")
# game.load_config("../../examples/config_all_actions/defend_the_center.cfg")
# game.load_config("../../examples/config_all_actions/defend_the_line.cfg")
# game.load_config("../../examples/config_all_actions/health_gathering.cfg")      # large with constant damage
# game.load_config("../../examples/config_all_actions/my_way_home.cfg")           # labirinth
game.load_config("../../examples/config_all_actions/predict_position.cfg")      # large map with 1 randomly moving mob
game.load_config("../../examples/config_all_actions/attempt1.cfg")      # large map with 1 randomly moving mobr
game.load_config("../../examples/config_all_actions/attempt2_map.cfg")      # large map with 1 randomly moving mobr
# game.load_config("../../examples/config_all_actions/take_cover.cfg")

# resolution = ScreenResolution.RES_1280X1024
# record = False

resolution = ScreenResolution.RES_160X120
record = True
output_folder = '../../data/free2/'

# print(dir(ScreenResolution))
# exit(0)

# Sets other rendering option
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

game.set_screen_resolution(resolution)

# Enables spectator mode, so you can play. Sounds strange but it is agent who is supposed to watch not you.
game.set_window_visible(True)

spectator = True
game.set_mode(Mode.PLAYER)
if spectator:
    game.set_mode(Mode.SPECTATOR)

game.init()
sleep_time = 40
episodes = 10
cl.init(record=record, output=output_folder, skip=9,  mode=resolution)
distance = 0


sum = 0
dsp = dp.Dispatcher()
dsp = DispatcherCircle.DispatcherCircle()
# dsp = DispatcherLine.DispatcherLine()
# dsp = radp.ActionDispatcher()

if spectator:
    dsp = DummyDispatcher.DummyDispatcher()


def printCross(depth):
    step = 5
    y = len(depth) / 2 + 2
    x = len(depth[0]) / 2 + 2
    depth[y, :] = 0
    depth[:, x] = 0
    depth[y, ::step] = 200
    depth[::step, x] = 200


for i in range(episodes):
    print("Episode #" +str(i+1))

    game.new_episode()
    while not game.is_episode_finished():

        s = game.get_state()
        img_buffer = s.image_buffer
        misc = s.game_variables

        last_action = dsp.action()
        if spectator:
            game.advance_action()
        else:
            game.make_action(last_action)

        last_action = game.get_last_action()
        last_reward = game.get_last_reward()

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
        depth = depth - 0
        dsp.handle(depth, last_action)

        # Display the image here!
        # printCross(depth) # print a cross
        cl.prntscr(depth, img, last_action)
        cv2.waitKey(sleep_time)
        depth *= 10 % 255 # make it brighter
        cv2.imshow('Doom Buffer', depth)



    print("episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
    sleep(1.0)

game.close()
