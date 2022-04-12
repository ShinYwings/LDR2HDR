import OpenEXR
import Imath
import numpy as np
import os
from datetime import datetime as dt
from numpy.random import uniform
import cv2

# If there is not input name, create the directory name with timestamp
def createNewDir(root_path, name=None):

    if name == None:
        print("[utils.py, createNewDir()] DirName is not defined in the arguments, define as timestamp")
        newpath = os.path.join(root_path, dt.now().strftime("%Y-%m-%d-%H:%M:%S"))
    else:
        newpath = os.path.join(root_path, name)

    """Create parent path if it doesn't exist"""
    if not os.path.isdir(newpath):
        os.mkdir(newpath)
    return newpath

def createTrainValidationDirpath(root_dir, createDir = False):
    
    if createDir == True:
        train_dir = createNewDir(root_dir, "train")
        val_dir = createNewDir(root_dir, "val")
    
    else:
        train_dir = os.path.join(root_dir, "train")
        val_dir = os.path.join(root_dir, "val") 

    return train_dir, val_dir


def writeHDR(arr, outfilename, imgshape):
    '''write HDR image using OpenEXR'''
    # Convert to strings
    R, G, B = [x.astype('float16').tostring() for x in [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]]

    im_height, im_width = imgshape

    HEADER = OpenEXR.Header(im_width, im_height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

    out = OpenEXR.OutputFile(outfilename, HEADER)
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()

def openexr2np(path):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    image = OpenEXR.InputFile(path)

    dw = image.header()['dataWindow']
    size = (dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1)
    (redstr, greenstr, bluestr) = image.channels("RGB",pt)
    
    red = np.frombuffer(redstr, dtype = np.float32)
    green = np.frombuffer(greenstr, dtype = np.float32)
    blue = np.frombuffer(bluestr, dtype = np.float32)

    for i in [red,green,blue]:
        i.shape=(size[1],size[0])

    red = np.expand_dims(red,axis=2)
    green = np.expand_dims(green, axis=2)
    blue = np.expand_dims(blue, axis=2)

    color = np.concatenate([red,green,blue],axis=2)

    return color