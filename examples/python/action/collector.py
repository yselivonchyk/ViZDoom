import cv2
import os, shutil
import json
import numpy as np

__output = '../../data//test/'
COLOR_SUBFOLDER = 'img'
DEPTH_SUBFOLDER = 'dep'
__resolution_subfolder = '0_0'
_counter = 0
_actual_count = 0
__toRecord = False
__skip = 0
__skip_same = True
auxilary_info = [0, 0, 0, 0, 0] # x, y, v_x, v_y, angle


def init(mode, record=True, output='../../data/test/', skip=0):
    print output
    global __resolution_subfolder, __toRecord, __skip, __output
    __toRecord = record
    __output = output
    __skip = skip
    __resolution_subfolder = str(mode).split('_')[1].replace('X', '_')

    _get_image_folders(True)
    print "ImageCollector init: %s, %d, %s" % (output, skip, str(record))


def start_recording():
    global __toRecord
    __toRecord = True

def _create_folder(path, clear_files=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if clear_files:
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print('file deletion issue: %s' % the_file, e)
    return path


def _get_image_folders(clear_files=False):
    """returns paths to folders for color images and depth images"""
    _create_folder(__output)
    suffix_folder = _create_folder(os.path.join(__output, COLOR_SUBFOLDER))
    color_folder = _create_folder(os.path.join(suffix_folder, __resolution_subfolder), clear_files=clear_files)
    suffix_folder = _create_folder(os.path.join(__output, DEPTH_SUBFOLDER))
    depth_folder = _create_folder(os.path.join(suffix_folder, __resolution_subfolder), clear_files=clear_files)
    return color_folder, depth_folder

action_file_name = 'action.txt'
actions = []
change_no_action = 0
image_changed = False

previous_action, previous_image = [], np.asarray([])


def register_action(action):
    global auxilary_info
    auxilary_info = [0, 0, 0, 0, auxilary_info[-1]]
    if action[3] != 0:
        auxilary_info[0] = 1
    if action[4] != 0:
        auxilary_info[0] = -1

    if action[5] != 0:
        auxilary_info[1] = -1
    if action[6] != 0:
        auxilary_info[1] = 1
    print(auxilary_info)

def prntscr(depth_img, img, action):
    global _counter, previous_action, _actual_count, \
        previous_image, change_no_action, image_changed

    _counter += 1

    any_action = sum([abs(x) for x in previous_action]) != 0
    image_changed = img.mean() != previous_image.mean()

    if __skip_same and not image_changed:
        previous_action, previous_image = action, img
        # print('skipped %d' % _counter)
        return

    _actual_count += 1
    with open(os.path.join(__output, action_file_name), 'w') as action_file:
        actions.append((_counter, str(image_changed), previous_action, auxilary_info[:]))
        json.dump(actions, action_file, indent=2)

    if any_action:
        change_no_action = 0
    else:
        if image_changed:
            change_no_action += 1
    # print('change_no_action: %d' % change_no_action, previous_action)

    color_output, depth_output = _get_image_folders()
    name = '_%05d.jpg' % _counter
    if __toRecord and _counter >= __skip:
        # print __output + 'dep' + name, depth_img
        if depth_img is not None:
            cv2.imwrite(os.path.join(depth_output, name), depth_img)
        if img is not None:
            cv2.imwrite(os.path.join(color_output, name), img)

    previous_action, previous_image = action, img


