import numpy as np

from skimage import transform, exposure, restoration, morphology
from structure_tensor import eig_special_2d, structure_tensor_2d
from tqdm import tqdm
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose


def detect_and_rotate_angle(video_frames, rho, sigma, num_frames_for_angle=4):
    """
    Detects the rotation angle of a video based on Hough line transform
    from a subset of frames and rotates all frames accordingly.

    Parameters:
    - video_frames: NumPy array representing the video frames with shape (z, x, y) or (z, c, x, y).
    - num_frames_for_angle: Number of frames to use for angle detection.

    Returns:
    - rotated_frames: NumPy array representing the rotated video frames.
    """
    video_frames = np.array(video_frames)

    if video_frames.ndim == 4:
        frames_for_angle = video_frames[:num_frames_for_angle, 0, :, :]
    else:
        frames_for_angle = video_frames[:num_frames_for_angle]
    average_angle = compute_average_angle(frames_for_angle, rho, sigma)
    rotated_frames = np.zeros_like(video_frames)

    for idx, frame in enumerate(video_frames):
        if video_frames.ndim == 4:
            rotated_frame = np.array(
                [transform.rotate(channel, angle=average_angle, preserve_range=True) for channel in frame])
        else:
            rotated_frame = transform.rotate(frame, angle=average_angle, preserve_range=True)

        rotated_frames[idx] = rotated_frame
    rotated_frames = rotated_frames.astype(np.float32)
    return rotated_frames


def angle_from_orientation(orientation):
    """
    Calculate the mean angle considering special cases.

    Parameters:
    - orientation (float): The orientation angle.

    Returns:
    - float: The mean angle.
    """
    # Ensure orientation is in the range [0, 180)
    orientation = orientation % 180
    angle = 0

    if 0 <= orientation < 90:
        if orientation >= 45:
            angle = 90 - orientation
        else:
            angle = -orientation + 90
    elif 90 <= orientation < 180:
        if orientation >= 135:
            angle = 270 - orientation
        else:
            angle = 90 - orientation
    return angle


def compute_average_angle(frames, sigma, rho):
    """
    Computes the average rotation angle based on Hough line transform.
    Then rotates the image to have horizontal grooves based on structure tensor orientation.

    Parameters:
    - frames: NumPy array representing the frames with shape (num_frames, x, y).

    Returns:
    - average_angle: The average rotation angle.
    """

    angles = []
    for frame in frames:
        a = structure_tensor_2d(frame.astype(np.float32), sigma=sigma, rho=rho)
        val, vec = eig_special_2d(a)
        ori = np.arctan2(vec[1], vec[0])
        median_ori = np.rad2deg(np.median(ori))
        angles.append(median_ori)

    orientation = np.mean(angles)
    final_angle = angle_from_orientation(orientation)

    print(orientation, final_angle)

    return final_angle


def per_channel_scaling(image):
    """
    Perform per-channel scaling to the range [0, 1].

    Parameters:
    - image: numpy array with shape (t, c, x, y)

    Returns:
    - scaled_image: numpy array with the same shape, but values scaled to [0, 1]
    """

    image = image.astype(np.float32)

    if image.ndim == 3:
        # 3D Image (t, x, y)
        min_vals = np.min(image, axis=(1, 2), keepdims=True)
        max_vals = np.max(image, axis=(1, 2), keepdims=True)

    elif image.ndim == 4:
        min_vals = np.min(image, axis=(2, 3), keepdims=True)
        max_vals = np.max(image, axis=(2, 3), keepdims=True)

    else:
        raise ValueError("Input image must be 3D (t, x, y) or 4D (t, c, x, y).")

    scaled_image = (image - min_vals) / (max_vals - min_vals + 1e-8)
    return scaled_image


def apply_clahe(image, clip_limit=0.01, channel_to_process=1, clahe_kernel_size=30):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing to a 3D or 4D image.

    Parameters:
    - image (numpy.ndarray): Input image (3D or 4D).
    - clip_limit (float): Clip limit for CLAHE.
    - channel_to_process (int): If the input is 4D, specify the channel to process.
    - clahe_kernel_size (int): Size of kernel is image_len // clahe_kernel_size

    Returns:
    - numpy.ndarray: CLAHE processed image.
    """

    if image.ndim == 3:
        # 3D Image (t, x, y)
        clahe_result = np.zeros_like(image)

        for t in tqdm(range(image.shape[0])):
            clahe_res = exposure.equalize_adapthist(image[t], clip_limit=clip_limit, kernel_size=(
            image.shape[1] // clahe_kernel_size, image.shape[2] // clahe_kernel_size))
            clahe_result[t] = clahe_res

    elif image.ndim == 4:
        # 4D Image (t, c, x, y)
        if channel_to_process is None:
            raise ValueError("For 4D images, specify the channel to process.")

        clahe_result = np.copy(image)
        for t in tqdm(range(image.shape[0])):
            # rescaled = exposure.adjust_log(image[t, channel_to_process], 1)
            clahe_res = exposure.equalize_adapthist(image[t, channel_to_process], clip_limit=clip_limit, kernel_size=(
            image.shape[2] // clahe_kernel_size, image.shape[3] // clahe_kernel_size))
            clahe_result[t, channel_to_process] = clahe_res
            del clahe_res

    else:
        raise ValueError("Input image must be 3D (t, x, y) or 4D (t, c, x, y).")

    return clahe_result


def apply_intensity_clipping_and_denoising(image, clip_percentile=1.5, weight=0.1, channel_to_process=1):
    """
    Apply intensity clipping and total variation denoising to a 3D or 4D image.

    Parameters:
    - image (numpy.ndarray): Input image (3D or 4D).
    - clip_percentile (float): Percentile value for intensity clipping.
    - weight (float): Weight parameter for total variation denoising.
    - channel_to_process (int): If the input is 4D, specify the channel to process.

    Returns:
    - numpy.ndarray: Processed image.
    """
    if image.ndim == 3:
        # 3D Image (t, x, y)
        processed_image = np.zeros_like(image)

        for t in tqdm(range(image.shape[0])):
            # Intensity clipping
            clip_max = np.percentile(image[t], 100 - clip_percentile)
            clipped_image = np.clip(image[t], 0, clip_max)
            # Total variation denoising
            denoised_image = restoration.denoise_tv_chambolle(clipped_image, weight=weight)
            denoised_image = exposure.adjust_sigmoid(denoised_image, cutoff=0.7, gain=5)
            processed_image[t] = denoised_image
            del clipped_image
            del denoised_image

    elif image.ndim == 4:
        # 4D Image (t, c, x, y)
        if channel_to_process is None:
            raise ValueError("For 4D images, specify the channel to process.")

        processed_image = np.copy(image)

        for t in tqdm(range(image.shape[0])):
            # Intensity clipping for the specified channel
            clip_max = np.percentile(image[t, channel_to_process], 100 - clip_percentile)
            clipped_image = np.clip(image[t, channel_to_process], 0, clip_max)

            # Total variation denoising for the specified channel
            denoised_image = restoration.denoise_tv_chambolle(clipped_image, weight=weight)
            denoised_image = exposure.adjust_sigmoid(denoised_image, cutoff=0.65, gain=5)
            processed_image[t, channel_to_process] = denoised_image
            del clipped_image
            del denoised_image
    else:
        raise ValueError("Input image must be 3D (t, x, y) or 4D (t, c, x, y).")

    return processed_image


def remove_thin_rows(image):
    """
    Removes thin rows of consecutive ones (white pixels) from a binary image.

    Parameters:
    - image (numpy.ndarray): Binary image represented as a 2D NumPy array.

    Returns:
    - numpy.ndarray: A modified binary image with thin consecutive rows of ones removed.
    """

    # Convert the list to a NumPy array
    arr = np.array(image[:, 1])

    # Find the indices where the values change from 0 to 1 and from 1 to 0
    indices_on = np.where(np.diff(np.concatenate(([0], arr, [0]))) == 1)[0]
    indices_off = np.where(np.diff(np.concatenate(([0], arr, [0]))) == -1)[0]

    # Calculate the consecutive ones lengths
    consecutive_ones = indices_off - indices_on

    # Change consecutive ones less than 2 to zeros
    for i, length in enumerate(consecutive_ones):
        if length < 2:
            arr[indices_on[i]:indices_off[i] + 1] = 0

    result_matrix = np.ones((len(arr), image.shape[1])) * arr[:, np.newaxis]

    return result_matrix


def filter_microgrooves(image):
    """
    Equalizes the average brightness of frames in a video.

    Parameters:
    - video_data (numpy.ndarray): Input video data with shape (t, c, x, y).
    - target_brightness (float): Target average brightness.

    Returns:
    - numpy.ndarray: Equalized video data.
    """
    thresholded_image = np.copy(image)
    zeros_channel = np.zeros_like(image[:, :1, :, :])
    thresholded_image = np.concatenate((thresholded_image, zeros_channel), axis=1)
    thresh = []
    for t in tqdm(range(image.shape[0])):
        frame = image[t, 0, :, :]
        frame[frame == 0] = np.nan
        mean_intensity_per_row = np.nanmedian(frame, axis=1)
        thresh.append(np.mean(mean_intensity_per_row))
    threshold = np.mean(thresh) + 0.03
    for t in tqdm(range(image.shape[0])):
        frame = image[t, 0, :, :]
        frame[frame == 0] = np.nan
        mean_intensity_per_row = np.nanmedian(frame, axis=1)
        binary_mask = mean_intensity_per_row <= threshold

        thresholded_image[t, 2, binary_mask] = 1  # Set pixels above threshold to 1 (white)

        thresholded_image[t, 2, :, :] = remove_thin_rows(thresholded_image[t, 2, :, :])

    return thresholded_image


def filter_microgrooves_with_model(image, model, device="cuda"):
    """
    Applies a deep learning model to filter microgrooves in a multi-channel image.

    Parameters:
    - image (numpy.ndarray): Multi-channel input image (e.g., RGB) represented as a NumPy array.
    - model (torch.nn.Module): Pre-trained deep learning model for microgroove detection.
    - device (str): Device on which to run the model (default is "cuda" for GPU).

    Returns:
    - numpy.ndarray: Processed image with filtered microgrooves.

    The function takes a multi-channel input image and a pre-trained deep learning model.
    It applies the model to detect microgrooves and updates the image accordingly.
    """

    post = Compose([AsDiscrete(threshold=0.5)])
    res_image = np.copy(image)
    zeros_channel = np.zeros_like(image[:, :1, :, :])
    res_image = np.concatenate((res_image, zeros_channel), axis=1)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = torch.tensor(image).to(device)
        for i in range(len(image)):
            roi_size = (512, 512)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(image[i, 0, :, :].unsqueeze(0).unsqueeze(0), roi_size, sw_batch_size, model)
            val_outputs = np.squeeze(post(val_outputs).detach().cpu())
            row_medians = np.median(val_outputs, axis=1)
            mask = row_medians == 1
            res_image[i, 2, mask, :] = 1
            # res_image[i, 2, :, :] = val_outputs
    torch.cuda.empty_cache()
    return res_image



