import argparse
import network
import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def pre_process_image(image):
    img = image.convert('RGB')
    img = img.resize((640, 480))

    return img

def post_process_segmentation(segmentation):
    # Clean up B/W image using morphological operations
    kernel = np.ones((15,15) , np.uint8) # kernel side determines resolution
    colorized_preds = cv.morphologyEx(segmentation, cv.MORPH_OPEN, kernel)
    colorized_preds = cv.morphologyEx(colorized_preds, cv.MORPH_CLOSE, kernel)

    colorized_preds = Image.fromarray(colorized_preds)
    
    # Medium filter to smooth edges
    colorized_preds = colorized_preds.filter(ImageFilter.ModeFilter(size=25))

    return colorized_preds
