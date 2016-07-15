import cv2
import os, shutil


output = 'output/'
__counter = 0
__toRecord = False


def init(record=True):
    global __toRecord
    __toRecord = record
    if not os.path.exists(output):
        os.makedirs(output)
    for the_file in os.listdir(output):
        file_path = os.path.join(output, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def prntscr(depth_img, img):
    global __counter
    __counter += 1
    name = output + '_%05d.jpg' % __counter
    if __toRecord:
        if depth_img is not None:
            cv2.imwrite('dep' + name, depth_img)
        if img is not None:
            cv2.imwrite('img' + name, img)
