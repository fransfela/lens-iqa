"""
Image I/O utilities for LENS
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import warnings


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR format) or None if failed
        
    Examples:
        >>> img = load_image('image.png')
        >>> print(img.shape)
        (480, 640, 3)
    """
    image_path = str(image_path)
    
    if not Path(image_path).exists():
        warnings.warn(f"Image file not found: {image_path}")
        return None
    
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            warnings.warn(f"Failed to load image: {image_path}")
            return None
        
        return img
    
    except Exception as e:
        warnings.warn(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save an image to file
    
    Args:
        image: Image as numpy array (BGR format)
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
        
    Examples:
        >>> img = load_image('input.png')
        >>> save_image(img, 'output.png')
        True
    """
    output_path = str(output_path)
    
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(output_path, image)
        
        if not success:
            warnings.warn(f"Failed to save image: {output_path}")
            return False
        
        return True
    
    except Exception as e:
        warnings.warn(f"Error saving image {output_path}: {e}")
        return False


def load_images_from_paths(image_paths: List[Union[str, Path]], 
                           verbose: bool = True) -> List[Optional[np.ndarray]]:
    """
    Load multiple images from a list of paths
    
    Args:
        image_paths: List of image file paths
        verbose: Print loading progress
        
    Returns:
        List of images (some may be None if loading failed)
        
    Examples:
        >>> paths = ['img1.png', 'img2.png', 'img3.png']
        >>> images = load_images_from_paths(paths)
        >>> print(f"Loaded {len([img for img in images if img is not None])} images")
    """
    images = []
    
    for i, path in enumerate(image_paths):
        if verbose:
            print(f"Loading image {i+1}/{len(image_paths)}: {Path(path).name}")
        
        img = load_image(path)
        images.append(img)
    
    if verbose:
        n_success = len([img for img in images if img is not None])
        print(f"✓ Successfully loaded {n_success}/{len(image_paths)} images")
    
    return images


def validate_image(image: np.ndarray) -> bool:
    """
    Validate if image array is valid
    
    Args:
        image: Image array to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> img = load_image('image.png')
        >>> validate_image(img)
        True
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    # Check if image has valid shape
    if len(image.shape) not in [2, 3]:
        return False
    
    # Check if image has valid dimensions
    if image.shape[0] == 0 or image.shape[1] == 0:
        return False
    
    # Check if color image has 3 channels
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    
    return True


def get_image_info(image: np.ndarray) -> dict:
    """
    Get information about an image
    
    Args:
        image: Image array
        
    Returns:
        Dictionary with image information
        
    Examples:
        >>> img = load_image('image.png')
        >>> info = get_image_info(img)
        >>> print(info)
        {'height': 480, 'width': 640, 'channels': 3, 'dtype': 'uint8'}
    """
    if not validate_image(image):
        return {}
    
    info = {
        'height': image.shape[0],
        'width': image.shape[1],
        'channels': image.shape[2] if len(image.shape) == 3 else 1,
        'dtype': str(image.dtype),
        'size': image.size,
        'min_value': float(image.min()),
        'max_value': float(image.max()),
        'mean_value': float(image.mean())
    }
    
    return info


def resize_image(image: np.ndarray, 
                width: Optional[int] = None, 
                height: Optional[int] = None,
                scale: Optional[float] = None,
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize an image
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        scale: Scale factor (optional)
        interpolation: Interpolation method (cv2.INTER_*)
        
    Returns:
        Resized image
        
    Examples:
        >>> img = load_image('image.png')
        >>> resized = resize_image(img, width=320, height=240)
        >>> resized = resize_image(img, scale=0.5)
    """
    if scale is not None:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
    
    if width is None and height is None:
        return image
    
    if width is None:
        aspect = image.shape[1] / image.shape[0]
        width = int(height * aspect)
    
    if height is None:
        aspect = image.shape[0] / image.shape[1]
        height = int(width * aspect)
    
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    
    return resized


def convert_color_space(image: np.ndarray, 
                       conversion: str = 'BGR2RGB') -> np.ndarray:
    """
    Convert image color space
    
    Args:
        image: Input image
        conversion: Color conversion code (e.g., 'BGR2RGB', 'BGR2GRAY')
        
    Returns:
        Converted image
        
    Examples:
        >>> img_bgr = load_image('image.png')
        >>> img_rgb = convert_color_space(img_bgr, 'BGR2RGB')
        >>> img_gray = convert_color_space(img_bgr, 'BGR2GRAY')
    """
    conversion_code = getattr(cv2, f'COLOR_{conversion}', None)
    
    if conversion_code is None:
        raise ValueError(f"Invalid conversion: {conversion}")
    
    converted = cv2.cvtColor(image, conversion_code)
    
    return converted