import numpy as np
import torch


def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target


def img_mask_to_np_input(img, mask, normalize=True):
    """
    Given an image and a mask, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    mask : torch.ByteTensor
        Binary matrix where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1] and pixel intensities to
        [-0.5, 0.5]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_points = mask[0].nonzero().size(0)
    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    nonzero_idx = mask.nonzero()
    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y = img[mask_img_size].view(batch_size, num_channels, num_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y = y.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x = (x - float(height) / 2) / (float(height) / 2)
        # Normalize y's to [-0.5, 0.5]
        y -= 0.5

    return x, y


def random_context_target_mask(img_size, num_context, num_extra_target):
    """Returns random context and target masks where 0 corresponds to a hidden
    value and 1 to a visible value. The visible pixels in the context mask are
    a subset of the ones in the target mask.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    _, height, width = img_size
    # Sample integers without replacement between 0 and the total number of
    # pixels. The measurements array will then contain pixel indices
    # corresponding to locations where pixels will be visible.
    measurements = np.random.choice(range(height * width),
                                    size=num_context + num_extra_target,
                                    replace=False)
    # Create empty masks
    context_mask = torch.zeros(width, height).byte()
    target_mask = torch.zeros(width, height).byte()
    # Update mask with measurements
    for i, m in enumerate(measurements):
        row = int(m / width)
        col = m % width
        target_mask[row, col] = 1
        if i < num_context:
            context_mask[row, col] = 1
    return context_mask, target_mask


def batch_context_target_mask(img_size, num_context, num_extra_target,
                              batch_size, repeat=False):
    """Returns bacth of context and target masks, where the visible pixels in
    the context mask are a subset of those in the target mask.

    Parameters
    ----------
    img_size : see random_context_target_mask

    num_context : see random_context_target_mask

    num_extra_target : see random_context_target_mask

    batch_size : int
        Number of masks to create.

    repeat : bool
        If True, repeats one mask across batch.
    """
    context_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    target_mask_batch = torch.zeros(batch_size, *img_size[1:]).byte()
    if repeat:
        context_mask, target_mask = random_context_target_mask(img_size,
                                                               num_context,
                                                               num_extra_target)
        for i in range(batch_size):
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    else:
        for i in range(batch_size):
            context_mask, target_mask = random_context_target_mask(img_size,
                                                                   num_context,
                                                                   num_extra_target)
            context_mask_batch[i] = context_mask
            target_mask_batch[i] = target_mask
    return context_mask_batch, target_mask_batch


def xy_to_img(x, y, img_size):
    """Given an x and y returned by a Neural Process, reconstruct image.
    Missing pixels will have a value of 0.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, 2) containing normalized indices.

    y : torch.Tensor
        Shape (batch_size, num_points, num_channels) where num_channels = 1 for
        grayscale and 3 for RGB, containing normalized pixel intensities.

    img_size : tuple of ints
        E.g. (1, 32, 32) for grayscale or (3, 64, 64) for RGB.
    """
    _, height, width = img_size
    batch_size, _, _ = x.size()
    # Unnormalize x and y
    x = x * float(height / 2) + float(height / 2)
    x = x.long()
    y += 0.5
    # Permute y so it matches order expected by image
    # (batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
    y = y.permute(0, 2, 1)
    # Initialize empty image
    img = torch.zeros((batch_size,) + img_size)
    for i in range(batch_size):
        img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :]
    return img


def inpaint(model, img, context_mask, device):
    """
    Given an image and a set of context points, the model samples pixel
    intensities for the remaining pixels in the image.

    Parameters
    ----------
    model : models.NeuralProcessImg instance

    img : torch.Tensor
        Shape (channels, height, width)

    context_mask : torch.Tensor
        Binary tensor where 1 corresponds to a visible pixel and 0 to an
        occluded pixel. Shape (height, width). Must have dtype=torch.uint8
        or similar. 

    device : torch.device
    """
    is_training = model.neural_process.training
    # For inpainting, use Neural Process in prediction mode
    model.neural_process.training = False
    target_mask = 1 - context_mask  # All pixels which are not in context
    # Add a batch dimension to tensors and move to GPU
    img_batch = img.unsqueeze(0).to(device)
    context_batch = context_mask.unsqueeze(0).to(device)
    target_batch = target_mask.unsqueeze(0).to(device)
    p_y_pred = model(img_batch, context_batch, target_batch)
    # Transform Neural Process output back to image
    x_target, _ = img_mask_to_np_input(img_batch, target_batch)
    # Use the mean (i.e. loc) parameter of normal distribution as predictions
    # for y_target
    img_rec = xy_to_img(x_target.cpu(), p_y_pred.loc.detach().cpu(), img.size())
    img_rec = img_rec[0]  # Remove batch dimension
    # Add context points back to image
    context_mask_img = context_mask.unsqueeze(0).repeat(3, 1, 1)
    img_rec[context_mask_img] = img[context_mask_img]
    # Reset model to mode it was in before inpainting
    model.neural_process.training = is_training
    return img_rec
