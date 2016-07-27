import cv2
import os, shutil


__output = '../../data//test/'
COLOR_SUBFOLDER = 'img'
DEPTH_SUBFOLDER = 'dep'
__resolution_subfolder = '0_0'
__counter = 0
__toRecord = False
__skip = 0


def init(mode, record=True, output='../../data//test/', skip=0):
    print output
    global __resolution_subfolder, __toRecord, __skip, __output
    __toRecord = record
    __output = output
    __skip = skip
    __resolution_subfolder = str(mode).split('_')[1].replace('X', '_')

    _get_image_folders(True)
    print "ImageCollector init: %s, %d, %s" % (output, skip, str(record))


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


def prntscr(depth_img, img):
    global __counter
    __counter += 1
    color_output, depth_output = _get_image_folders()
    name = '_%05d.jpg' % __counter
    if __toRecord and __counter >= __skip:
        # print __output + 'dep' + name, depth_img
        if depth_img is not None:
            cv2.imwrite(os.path.join(depth_output, name), depth_img)
        if img is not None:
            cv2.imwrite(os.path.join(color_output, name), img)

