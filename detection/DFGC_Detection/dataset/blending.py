# borrow from https://github.com/neverUseThisName/Face-X-Ray
import numpy as np
import random
from PIL import Image
from imgaug import augmenters as iaa
from DeepFakeMask import dfl_full, facehull, components,extended
import cv2

distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])

def random_get_hull(landmark,img1):
    hull_type = random.choice([0,1,2,3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255

def random_erode_dilate(mask, ksize=None):
    if random.random()>0.5:
        if ksize is  None:
            ksize = random.randint(1,21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask,kernel,1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.dilate(mask,kernel,1)/255
    return mask

def read_png_or_jpg(image):
    if random.random() > 0.75:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        quality = random.randint(75, 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        face_img_encode = cv2.imencode('.jpg', image, encode_param)[1]
        image = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image	

# borrow from https://github.com/MarekKowalski/FaceSwap
# def blendImages(src, dst, mask, featherAmount=0.2):
   
#     maskIndices = np.where(mask != 0)
    
#     src_mask = np.ones_like(mask)
#     dst_mask = np.zeros_like(mask)

#     maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
#     faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
#     featherAmount = featherAmount * np.max(faceSize)

#     hull = cv2.convexHull(maskPts)
#     dists = np.zeros(maskPts.shape[0])
#     for i in range(maskPts.shape[0]):
#         dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

#     weights = np.clip(dists / featherAmount, 0, 1)

#     composedImg = np.copy(dst)
#     composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

#     composedMask = np.copy(dst_mask)
#     composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
#                 1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

#     return composedImg, composedMask
import numpy as np
import cv2

def blendImages(src, dst, mask, featherAmount=0.2):
    """
    Blend a foreground image src onto a background image dst using mask and feathering.
    Supports multi-channel (e.g. RGB) images.

    Args:
        src (np.ndarray): Foreground image, shape (H, W, C).
        dst (np.ndarray): Background image, shape (H, W, C).
        mask (np.ndarray): Single-channel mask or multi-channel mask, non-zero indicates blending region.
        featherAmount (float): Relative feathering factor (0-1) multiplied by face size.

    Returns:
        composedImg (np.ndarray): Blended color image, shape (H, W, C).
        composedMask (np.ndarray): Blended single-channel mask, shape (H, W).
    """
    # 0. Ensure mask is single-channel 2D
    if mask.ndim == 3:
        # if mask has channels, collapse to single channel
        # e.g., take first channel or any-channel non-zero
        mask = mask[:, :, 0]

    # 1. Compute mask indices (row, col) of non-zero mask pixels
    maskIndices = np.where(mask != 0)
    ri, ci = maskIndices

    # 2. Build point array of shape (N, 2) and convert to int32
    maskPts = np.stack((ci, ri), axis=1).astype(np.int32)

    # 3. Compute feather radius based on mask bounding box
    faceSize = maskPts.max(axis=0) - maskPts.min(axis=0)
    feather_px = featherAmount * max(faceSize)

    # 4. Compute convex hull of mask points
    hull = cv2.convexHull(maskPts)

    # 5. Compute signed distance of each mask point to hull
    dists = np.zeros(len(maskPts), dtype=np.float32)
    for i, (x, y) in enumerate(maskPts):
        dists[i] = cv2.pointPolygonTest(hull, (int(x), int(y)), True)

    # 6. Normalize distances into blending weights [0,1]
    weights = np.clip(dists / feather_px, 0.0, 1.0).astype(np.float32)

    # 7. Prepare output images
    composedImg = dst.copy().astype(np.float32)
    composedMask = np.zeros_like(mask, dtype=np.float32)

    # 8. Extract pixel values for blending
    src_vals = src[ri, ci, :].astype(np.float32)
    dst_vals = dst[ri, ci, :].astype(np.float32)

    # 9. Blend each channel using weights
    w = weights[:, None]       # shape (N, 1)
    blended_vals = w * src_vals + (1.0 - w) * dst_vals  # shape (N, C)

    # 10. Assign blended pixels back to composed image
    composedImg[ri, ci, :] = blended_vals

    # 11. Create composed single-channel mask
    composedMask[ri, ci] = weights

    # 12. Convert composedImg back to original dtype (e.g., uint8)
    composedImg = np.clip(composedImg, 0, 255).astype(dst.dtype)

    return composedImg, composedMask



# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    
    maskIndices = np.where(mask != 0)
    
    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst

def blend_fake_real_img(this_path, this_landmark, searched_path, searched_landmark, size=(256, 256)):
    foreground_face = cv2.cvtColor(cv2.imread(this_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    background_face = cv2.cvtColor(cv2.imread(searched_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    w = random.randint(256, 300)
    size = (w, w)
    foreground_face = cv2.resize(foreground_face, size)
    background_face = cv2.resize(background_face, size)

    # this_landmark = this_landmark * (size[0]/256)
    # get random type of initial blending mask    
    mask = random_get_hull(this_landmark, foreground_face)
    
    #  random deform mask
    mask = distortion.augment_image(mask)
    mask = random_erode_dilate(mask)
    
    # apply color transfer
    foreground_face = colorTransfer(background_face, foreground_face, mask*255)
    
    # blend two face
    blended_face, mask = blendImages(foreground_face, background_face, mask*255)
    blended_face = blended_face.astype(np.uint8)
    blended_face = read_png_or_jpg(blended_face)


    return blended_face
