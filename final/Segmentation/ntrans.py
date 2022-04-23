import SimpleITK as sitk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import scipy.misc
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from os.path import join
import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from os.path import join
import os
import numpy as np
import functools
from multiprocessing import Pool

offset = './data/'
types = ['train', 'test']

p = Pool(16)

from osing import singp

def pfunc():
  for tp in types:
    path = join(offset, tp)
    names = [f for f in os.listdir(path) if f.endswith('.mhd')]
    p.map(functools.partial(singp, path=path, mode=tp, offset=offset), names)

if __name__=='__main__':
    pfunc()


