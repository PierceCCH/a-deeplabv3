from tqdm import tqdm
import network
import utils
import os

from datasets import Cityscapes

import torch
import torch.nn as nn
from torchvision import transforms as T

from PIL import Image
from glob import glob

import inference_utils as inf_utils
import calc_steering_angle as st_util

def main():
    opts = inf_utils.get_argparser().parse_args()
    opts.dataset.lower() == 'cityscapes'
    opts.num_classes = 19
    decode_fn = Cityscapes.decode_target

    device = torch.device('cpu')
    print("Device: %s" % device)

    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    # Image dataloader, will be replaced by video stream
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain: Your weights are not found.")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
        
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            
            image = Image.open(img_path).convert('RGB')
            image = inf_utils.pre_process_image(image)
            image_tensor = transform(image).unsqueeze(0)
            
            pred = model(image_tensor).max(1)[1].cpu().numpy()[0]
            colorized_preds = decode_fn(pred).astype('uint8')

            colorized_preds = inf_utils.post_process_segmentation(colorized_preds)

            # Create instance of steering angle calculator
            lane_follower = st_util.HandCodedLaneFollower(img_name)
            pred_with_angle = lane_follower.follow_lane(colorized_preds)

            # Save image to file
            img_with_angle = Image.fromarray(pred_with_angle)
            final_img = Image.blend(img_with_angle, image, 0.25)
            if opts.save_val_results_to:
                final_img.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

if __name__ == '__main__':
    main()
