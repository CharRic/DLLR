import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from SelfCorrectionHumanParsing import networks
from SelfCorrectionHumanParsing.datasets.simple_extractor_dataset import SimpleFolderDataset
from SelfCorrectionHumanParsing.utils.transforms import transform_logits

# The code is implemented based on SelfCorrectionHumanParsing.simple_extractor.py

dataset_settings = {'lip': {'input_size': [473, 473], 'num_classes': 20,
    'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants',
              'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
              'Right-shoe']}, 'atr': {'input_size': [512, 512], 'num_classes': 18,
    'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
              'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']},
    'pascal': {'input_size': [512, 512], 'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'], }}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='pascal', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='SelfCorrectionHumanParsing/checkpoints/final.pth',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='./data/sysu', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='./data/sysu_mask', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def main():
    print("initalize setting.")
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    print("Loading model.")

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
    # download from SCHP pretrained model.
    if not os.path.isfile(args.model_restore):
        if args.dataset == 'lip':
            url = 'https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH'
        elif args.dataset == 'atr':
            url = 'https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP'
        elif args.dataset == 'pascal':
            url = 'https://drive.google.com/uc?id=1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE'

        output = args.model_restore
        gdown.download(url, output, quiet=False)
    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    print("Start.")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])
    palette = [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    cams = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
    for cam in cams:
        cam_path = os.path.join(args.input_dir, cam)
        cam_out_path = os.path.join(args.output_dir, cam)
        img_dirs = os.listdir(cam_path)
        print("Processing {}.".format(cam))
        for img_dir in tqdm(img_dirs):
            input_dir = os.path.join(cam_path, img_dir)
            output_dir = os.path.join(cam_out_path, img_dir)
            dataset = SimpleFolderDataset(root=input_dir, input_size=input_size, transform=transform)
            dataloader = DataLoader(dataset)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # palette = get_palette(num_classes)

            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    image, meta = batch
                    img_name = meta['name'][0]
                    c = meta['center'].numpy()[0]
                    s = meta['scale'].numpy()[0]
                    w = meta['width'].numpy()[0]
                    h = meta['height'].numpy()[0]

                    output = model(image.cuda())
                    upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                    upsample_output = upsample_output.squeeze()
                    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                    logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h,
                                                     input_size=input_size)
                    parsing_result = np.argmax(logits_result, axis=2)
                    parsing_result_path = os.path.join(output_dir, img_name[:-4] + '.png')
                    output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                    output_img.putpalette(palette)
                    output_img.save(parsing_result_path)
                    if args.logits:
                        logits_result_path = os.path.join(output_dir, img_name[:-4] + '.npy')
                        np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()
