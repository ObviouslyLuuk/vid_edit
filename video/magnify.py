
from torch import tensor
from torch.nn.functional import conv2d
from torchvision import transforms
from PIL import Image
from cv2 import getGaussianKernel
import numpy as np

from warp import warp_images


def magnify_tensor(img, point, factor):
    """Magnify the image tensor by a factor, keeping the 
    reference point in the same place"""
    assert len(img.shape) == 4, 'Input tensor must have shape (B, H, W, C)'
    h, w = img.shape[1:3]
    img = tensor(img).permute(0, 3, 1, 2) # (B, C, H, W)
    magnified_img = transforms.functional.resize(img, 
        (int(h*factor), int(w*factor)), interpolation=Image.BILINEAR,
        antialias=True).permute(0, 2, 3, 1).numpy() # (B, H, W, C)
    if isinstance(point, tuple):
        point = np.array(point)[None, :] # Convert to tensor shape (1,2)
    else:
        assert point.shape[0] == img.shape[0], 'point must be tensor of shape (B,2)'
    # Adjust point for magnification
    magnified_point = (point*factor).astype(int)
    # Get upper left corner in magnified coordinates
    UL_corner = magnified_point - point

    # Crop to the original size, with the given point staying in the same place
    # Potentially with a different point for each image in the batch
    batch_indices = np.arange(img.shape[0])[:, None, None]
    row_indices = np.arange(h)[None, :, None]
    col_indices = np.arange(w)[None, None, :]
    row_indices = row_indices + UL_corner[:, 1][:, None, None]
    col_indices = col_indices + UL_corner[:, 0][:, None, None]
    magnified_img = magnified_img[batch_indices, row_indices, col_indices, :]
    return magnified_img


###############################################################################
# Masks
###############################################################################

def get_circle_mask_tensor(h, w, center, radius):
    """Get a tensor mask of (B, H, W) with a circle of radius at center,
    where the masked area is 1 and the rest is 0.
    center is a tensor of shape (B, 2), radius is (B,)"""
    assert len(center.shape) == 2, 'center must be tensor of shape (B,2)'
    assert len(radius.shape) == 1, 'radius must be tensor of shape (B,)'
    assert center.shape[0] == radius.shape[0], 'center and radius must have the same batch size'
    row_indices = np.arange(h)[None, :, None]
    col_indices = np.arange(w)[None, None, :]
    center = center[:, None, None, :]
    radius = radius[:, None, None]
    mask = np.sqrt(
            (row_indices - center[:, :, :, 1])**2 +\
            (col_indices - center[:, :, :, 0])**2
        ) < radius[:, :, :]
    return mask.astype(float)

def get_oval_mask_tensor(h, w, center, size, angle):
    """Get a tensor mask of (B, H, W) with an oval of size at center,
    where the masked area is 1 and the rest is 0.
    center is a tensor of shape (B, 2), size is (B, 2), angle is (B,)"""
    assert len(center.shape) == 2, 'center must be tensor of shape (B,2)'
    assert len(size.shape) == 2, 'size must be tensor of shape (B,2)'
    assert len(angle.shape) == 1, 'angle must be tensor of shape (B,)'
    assert center.shape[0] == size.shape[0] == angle.shape[0], 'center, size and angle must have the same batch size'
    row_indices = np.arange(h)[None, :, None]
    col_indices = np.arange(w)[None, None, :]
    center = center[:, None, None, :]
    size = size[:, None, None, :] // 2 # We use radius
    angle = angle[:, None, None]
    angle_rad = angle * np.pi / 180
    mask = (
            (row_indices - center[:, :, :, 1])*np.cos(angle_rad) +\
            (col_indices - center[:, :, :, 0])*np.sin(angle_rad)
        )**2 / size[:, :, :, 0]**2 + (
            (row_indices - center[:, :, :, 1])*np.sin(angle_rad) -\
            (col_indices - center[:, :, :, 0])*np.cos(angle_rad)
        )**2 / size[:, :, :, 1]**2 < 1
    return mask.astype(float)

def get_rectangle_mask_tensor(h, w, center, size, angle):
    """Get a tensor mask of (B, H, W) with a rectangle of size at center,
    where the masked area is 1 and the rest is 0.
    center is a tensor of shape (B, 2), size is (B, 2), angle is (B,)"""
    assert len(center.shape) == 2, 'center must be tensor of shape (B,2)'
    assert len(size.shape) == 2, 'size must be tensor of shape (B,2)'
    assert len(angle.shape) == 1, 'angle must be tensor of shape (B,)'
    assert center.shape[0] == size.shape[0] == angle.shape[0], 'center, size and angle must have the same batch size'
    row_indices = np.arange(h)[None, :, None]
    col_indices = np.arange(w)[None, None, :]
    center = center[:, None, None, :]
    size = size[:, None, None, :]
    angle = angle[:, None, None]
    # First get rectangle with sides parallel to the axes
    angle_rad = angle * np.pi / 180
    mask_height = (
            np.abs((row_indices - center[:, :, :, 1])*np.cos(angle_rad) +\
            (col_indices - center[:, :, :, 0])*np.sin(angle_rad))
        ) < size[:, :, :, 0] / 2
    mask_width = (
            np.abs((row_indices - center[:, :, :, 1])*np.sin(angle_rad) -\
            (col_indices - center[:, :, :, 0])*np.cos(angle_rad))
        ) < size[:, :, :, 1] / 2
    mask = mask_height * mask_width
    return mask.astype(float)

def get_pill_mask_tensor(h, w, center, size, angle):
    """Get a tensor mask of (B, H, W) with a pill of size at center,
    where the masked area is 1 and the rest is 0.
    center is a tensor of shape (B, 2), size is (B, 2), angle is (B,)"""
    assert len(center.shape) == 2, 'center must be tensor of shape (B,2)'
    assert len(size.shape) == 2, 'size must be tensor of shape (B,2)'
    assert len(angle.shape) == 1, 'angle must be tensor of shape (B,)'
    assert center.shape[0] == size.shape[0] == angle.shape[0], 'center, size and angle must have the same batch size'
    
    mask = get_rectangle_mask_tensor(h, w, center, size, angle)

    # Add the two circles at the ends, using the get_circle_mask_tensor function
    angle_rad = angle * np.pi / 180
    # We want the vector pointing from the center to the tip of the pill
    # This should be of shape (B, 2)
    tip_vector = np.array([np.sin(angle_rad), np.cos(angle_rad)]).T # this is reversed sin,cos because we're transposing?
    circle_center1 = center - tip_vector * size[:, 0] / 2
    circle_center2 = center + tip_vector * size[:, 0] / 2
    radius = size[:, 1] / 2
    circle_mask1 = get_circle_mask_tensor(h, w, circle_center1, radius)
    circle_mask2 = get_circle_mask_tensor(h, w, circle_center2, radius)
    
    mask = mask + circle_mask1 + circle_mask2
    mask = mask.clip(0, 1)
    return mask.astype(float)

###############################################################################

def feather_edges(mask, feather_size:int):
    """Feather the edges of the mask by feather_size pixels"""
    assert len(mask.shape) == 3, 'mask must be tensor of shape (B,H,W)'

    # Use a 1d gaussian kernel separably to feather the edges
    kernel_size = int(2 * feather_size + 1)
    kernel = tensor(getGaussianKernel(kernel_size, 0))[None, None] # (1, 1, kernel_size, 1)

    mask = tensor(mask)[:,None]
    mask = conv2d(mask, kernel, stride=1, padding=(feather_size, 0))
    kernel = kernel.permute(0, 1, 3, 2)
    mask = conv2d(mask, kernel, stride=1, padding=(0, feather_size))
    mask = mask.squeeze(1).numpy()
    return mask

def img_mask_blend(img1, img2, mask):
    """Blend two image tensors using a mask"""
    assert len(img1.shape) == 4, 'img1 must be tensor of shape (B, H, W, C)'
    assert len(img2.shape) == 4, 'img2 must be tensor of shape (B, H, W, C)'
    assert len(mask.shape) == 3, 'mask must be tensor of shape (B, H, W)'
    assert img1.shape == img2.shape, 'img1 and img2 must have the same shape'
    assert img1.shape[:3] == mask.shape, 'img1 and mask must have the same B, H, W'
    img1 = tensor(img1).permute(0, 3, 1, 2) # (B, C, H, W)
    img2 = tensor(img2).permute(0, 3, 1, 2) # (B, C, H, W)
    mask = tensor(mask).unsqueeze(1) # (B, 1, H, W)
    img = img1 * (1-mask) + img2 * mask
    return img.permute(0, 2, 3, 1).numpy() # (B, H, W, C)

def magnify_blend(img, point, factor, feather_size, mask, magnify_mode="uniform"):
    """Magnify the image tensor by a factor, keeping the 
    reference point in the same place, and blending with a mask.
    Magnify mode can be "uniform" or "warp"
    """
    if magnify_mode == "uniform":
        magnified_img = magnify_tensor(img, point, factor)
    elif magnify_mode == "warp":
        magnified_img = warp_images(img, point, factor)
    else:
        raise ValueError('magnify_mode must be "uniform" or "warp"')
    mask = feather_edges(mask, feather_size)
    img = img_mask_blend(img, magnified_img, mask)
    return img



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test with checkerboard image
    img = np.zeros((256, 256, 3))
    for i in range(16):
        for j in range(16):
            img[i::32, j::32] = [1., 1., 1.]
    img = img[None, :, :, :]
    # plt.imshow(img[0])
    # plt.show()

    # # Test magnification
    # point = np.array([[128, 128]])
    # factor = 2
    # magnified_img = magnify_tensor(img, point, factor)
    # plt.imshow(magnified_img[0])
    # plt.show()

    # # Test circle mask
    # center = np.array([[128, 128]])
    # radius = np.array([64])
    # mask = get_circle_mask_tensor(256, 256, center, radius)
    # plt.imshow(mask[0])
    # plt.show()

    # # Test oval mask
    # center = np.array([[128, 128]])
    # size = np.array([[64, 128]])
    # angle = np.array([10])
    # mask = get_oval_mask_tensor(256, 256, center, size, angle)
    # plt.imshow(mask[0])
    # plt.show()

    # # Test rectangle mask
    # center = np.array([[128, 128]])
    # size = np.array([[64, 128]])
    # angle = np.array([10])
    # mask = get_rectangle_mask_tensor(256, 256, center, size, angle)
    # plt.imshow(mask[0])
    # plt.show()

    # # Test pill mask
    # center = np.array([[128, 128]])
    # size = np.array([[128, 64]])
    # angle = np.array([10])
    # mask = get_pill_mask_tensor(256, 256, center, size, angle)
    # plt.imshow(mask[0])
    # plt.show()

    # # Test feathering
    # feather_size = 20
    # mask = feather_edges(mask, feather_size)
    # plt.imshow(mask[0])
    # plt.show()

    # # Test blending of img and magnified_img
    # img = img_mask_blend(img, magnified_img, mask)
    # plt.imshow(img[0])
    # plt.show()

    # Test magnify_blend
    point = np.array([[128, 128]])
    size = np.array([[128, 64]])
    angle = np.array([10])
    mask = get_pill_mask_tensor(*img.shape[1:3], center=point, 
        size=size, angle=angle)
    factor = 1.3
    feather_size = 20
    magnified = magnify_blend(img, point, factor, feather_size, mask)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(magnified[0])

    factor = 1.2
    warped = magnify_blend(img, point, factor, feather_size, mask, 
            magnify_mode="warp")
    
    ax[1].imshow(warped[0])
    plt.show()
