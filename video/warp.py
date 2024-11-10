
import numpy as np

# def warp_image(img, point, factor):
#     """Return the image warped by factor, from point as the reference, cropped to original size"""
#     h, w = img.shape[:2]
#     # Create a grid of coordinates
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     # Convert to float32
#     x = x.astype(np.float32)
#     y = y.astype(np.float32)
#     # Normalize to -1 to 1, with the given point as the origin
#     x = (x - point[0]) / w # shape: (h, w)
#     y = (y - point[1]) / h # shape: (h, w)
#     # Calculate the distance from the point
#     r = np.sqrt(x**2 + y**2) # Distance from point shape: (h, w)
#     # Calculate the new distance from the point
#     r = r ** (factor-1)
#     # Calculate the new coordinates
#     x = x * r * w + point[0]
#     y = y * r * h + point[1]

#     warped = cv2.remap(img, x, y, cv2.INTER_LINEAR)
#     return warped

def remap_images(image_tensor, x, y):
    """Return the images remapped to the new coordinates, cropped to original size.
    @param image_tensor: shape (B, H, W, C)
    @param x: shape (B, H, W)
    @param y: shape (B, H, W)
    @return remapped: shape (B, H, W, C)
    """
    h, w = image_tensor.shape[1:3]
    # For interpolation, we need the 4 nearest points
    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1
    # We also need to store the weights for interpolation
    x0_weight = x1 - x
    x1_weight = x - x0
    y0_weight = y1 - y
    y1_weight = y - y0
    # Clamp the values to the image size
    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)

    # Get advanced indexing for the 4 nearest points
    batch_indices = np.arange(image_tensor.shape[0])[:, None, None, None]
    y0 = y0[:, :, :, None]
    x0 = x0[:, :, :, None]
    y1 = y1[:, :, :, None]
    x1 = x1[:, :, :, None]
    channel_indices = np.arange(image_tensor.shape[3])[None, None, None, :]
    # Get the 4 nearest points
    Ia = image_tensor[batch_indices, y0, x0, channel_indices]
    Ib = image_tensor[batch_indices, y1, x0, channel_indices]
    Ic = image_tensor[batch_indices, y0, x1, channel_indices]
    Id = image_tensor[batch_indices, y1, x1, channel_indices]
    # Interpolate the values
    wa = (x0_weight * y0_weight)[:, :, :, None]
    wb = (x0_weight * y1_weight)[:, :, :, None]
    wc = (x1_weight * y0_weight)[:, :, :, None]
    wd = (x1_weight * y1_weight)[:, :, :, None]
    # Sum the values
    remapped = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return remapped

def warp_images(image_tensor, point, factor):
    """Return the images warped by factor, from point as the reference, 
    cropped to original size.
    @param image_tensor: shape (B, H, W, C)
    @param point: shape (B, 2)
    @param factor: float
    @return warped: shape (B, H, W, C)
    """
    h, w = image_tensor.shape[1:3]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)[None] # shape: (1, h, w)
    y = y.astype(np.float32)[None] # shape: (1, h, w)
    point = point[:, :, None, None] # shape: (B, 2, 1, 1)
    # Normalize to -1 to 1, with the given point as the origin
    x = (x - point[:, 0]) / w # shape: (B, h, w)
    y = (y - point[:, 1]) / h # shape: (B, h, w)
    # Calculate the distance from the point
    r = np.sqrt(x**2 + y**2) # Distance from point shape: (B, h, w)
    # Calculate the new distance from the point
    r = r ** (factor-1)
    # Calculate the new coordinates
    x = x * r * w + point[:, 0]
    y = y * r * h + point[:, 1]
    # Remap the images
    warped = remap_images(image_tensor, x, y)
    return warped


# TODO: add option for line instead of point



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test with checkerboard image
    img = np.zeros((256, 256, 3))
    for i in range(16):
        for j in range(16):
            img[i::32, j::32] = [1., 1., 1.]
    imgs = np.stack([img, img], axis=0)

    # # Warp the image
    # center = (128, 128)
    # factor = 2
    # warped = warp_image(img[0], center, factor)

    # Warp the image(s)
    center = np.array([[128, 128], [200, 200]])
    factor = 1.3
    warped = warp_images(imgs, center, factor)
    
    # Plot the images
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(warped[0])
    ax[1].imshow(warped[1])
    plt.show()

