import cv2
import os, shutil


__output = 'output/'
__counter = 0
__toRecord = False
__skip = 0


def init(record=True, output='output/', skip=0):
    global __toRecord, __skip, __output
    __toRecord = record
    __output = output
    __skip = skip
    print "ImageCollector init: %s, %d, %s" % (output, skip, str(record))
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
    name = '_%05d.jpg' % __counter
    if __toRecord and __counter >= __skip:
        print __output + 'dep' + name, depth_img
        if depth_img is not None:
            cv2.imwrite(__output + 'dep' + name, depth_img)
        if img is not None:
            cv2.imwrite(__output + 'img' + name, img)
