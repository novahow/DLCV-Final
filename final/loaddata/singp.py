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

def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = (im < -600).astype('int')
    # print(binary)
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)

    # print('1', np.unique(binary)) 
    '''
        Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    # print('2', np.unique(cleared)) 
    '''
    
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    # print('3', np.unique(label_image)) 
      
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 

    # print('4', np.unique(binary)) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)

    # print('5', np.unique(binary))  
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 

    # print('6', np.unique(binary)) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 

    # print('7', np.unique(binary))
    binary = binary.astype('int')
    binary[binary == 1] = 3
    binary[binary == 0] = 0
    return binary

def segment_lung_from_ct_scan(ct_scan):
    return np.array([get_segmented_lungs(slice) for slice in ct_scan])

def singp(n, path, mode, offset):
    print(n)
    mhd = sitk.ReadImage(join(path, n))
    dar = sitk.GetArrayFromImage(mhd)
    slc = np.copy(dar)
    slc[slc < -1000] = -1000
    slc[slc > 604] = 604 
    aseg = segment_lung_from_ct_scan(slc)
    # print(type(aseg), aseg.shape)
    # print(mhd.GetSize())
    # ss = mhd.GetSize()
    print('segged')
    aseg = aseg.astype('short')
    img = sitk.GetImageFromArray(aseg)
    img.SetSpacing(mhd.GetSpacing())
    img.SetOrigin(mhd.GetOrigin())
    img.SetDirection(mhd.GetDirection())
    # print(img.GetSpacing(), img.GetSize())
    # sitk.WriteImage(img, 'no_compression.mhd')
    print('imgok')
    print(join(join(join(offset, 'seg'), mode), n))
    sitk.WriteImage(img, join(join(join(offset, 'seg'), mode), n), useCompression = True)