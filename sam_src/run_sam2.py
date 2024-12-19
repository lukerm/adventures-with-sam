import argparse
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import hydra
from hydra.core.hydra_config import HydraConfig

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

ALPHA = 0.5
DEVICE = 'cpu'


def create_plain_image(colour: tuple) -> np.ndarray:
    overlay_array = np.zeros(shape=img.shape, dtype=np.uint8)
    for channel in range(3):
        overlay_array[:, :, channel] = int(colour[channel] * 255)

    return overlay_array


def create_segment_image(mask: np.ndarray, save_img_fname: str = None, save_dir: str = None) -> Image:
    with Image.fromarray(img) as img_pil:
        alpha_mask = Image.fromarray(mask).convert('L')
        img_pil.putalpha(alpha_mask)

        if save_img_fname:
            img_pil.save(os.path.join(save_dir, save_img_fname))

    return img_pil


def create_coloured_segment_overlay_image(mask: np.ndarray, colour: tuple, alpha: float, save_img_fname: str = None, save_dir: str = None) -> Image:
    overlay_array = create_plain_image(colour=colour)
    overlay = Image.fromarray(overlay_array)
    alpha_mask = Image.fromarray(mask * alpha * 255).convert('L')
    overlay.putalpha(alpha_mask)

    if save_img_fname:
        overlay.save(os.path.join(save_dir, save_img_fname))

    return overlay


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run SAM2 on a directory of images to produce overlays')
    parser.add_argument(
        '--img-dir', '-d', type=str, required=True, help='Directory containing images to process')
    parser.add_argument(
        '--model-type', '-m', type=str, default='tiny', choices=['tiny', 'small', 'base_plus', 'large'],
        help='Model size to use'
    )
    parser.add_argument(
        '--model-checkpoint-dir', type=str, default=os.path.expanduser('~/sam2/checkpoints'),
        help='Directory where model checkpoints were downloaded',
    )
    parser.add_argument(
        '--save-segment-imgs', action='store_true', help='Save the individual segment images in a subdirectory',
    )
    args = parser.parse_args()

    # Note: these constants are linked by the model size: e.g. '_tiny' in checkpoint, '_t' in config
    #       Also, "base_plus" model type translates to "b+"
    model_checkpoint_fname = f'sam2.1_hiera_{args.model_type}.pt'
    model_config_fname = f'sam2.1_hiera_{args.model_type[0]}{"+" if "plus" in args.model_type else ""}.yaml'
    img_dir = args.img_dir


    # Clear any previous Hydra state and initialize with relative config path
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="sam2", job_name="run_sam2", version_base="1.1")

    sam2_model = build_sam2(
        model_config_fname,
        os.path.join(args.model_checkpoint_dir, model_checkpoint_fname),
        device=DEVICE
    )
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)


    imgs_todo = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg')]
    for i_img, img_fname in enumerate(imgs_todo):
        t0 = datetime.now()
        print(f'Processing {i_img} / {len(imgs_todo)} images ({img_fname})')

        img_fname_stub = os.path.splitext(img_fname)[0]
        my_save_dir = os.path.join(img_dir, f'{img_fname_stub}_{args.model_type}')

        if args.save_segment_imgs:
            os.makedirs(my_save_dir, exist_ok=True)

        img = cv2.imread(os.path.join(img_dir, img_fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(img)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)  # sort by area

        col_map = plt.cm.get_cmap('hsv', len(masks))
        rs = np.random.RandomState(seed=i_img)
        randomizer = rs.choice(range(len(masks)), len(masks), replace=False)

        # Collector for all detected segments
        object_overlays = []

        for i_mask, mask in enumerate(masks):

            if args.save_segment_imgs:
                _ = create_segment_image(
                    mask=mask['segmentation'],
                    save_img_fname=f'{img_fname_stub}_mask_{i_mask}.png',
                    save_dir=my_save_dir,
                )

            colour = col_map(randomizer[i_mask])
            overlay = create_coloured_segment_overlay_image(
                mask=mask['segmentation'], alpha=ALPHA, colour=colour,
                save_img_fname=f'{img_fname_stub}_mask_{i_mask}_overlay.png' if args.save_segment_imgs else None,
                save_dir=my_save_dir
            )
            object_overlays.append(overlay)

        with Image.fromarray(img) as img_pil:
            for overlay in object_overlays:
                img_pil.paste(overlay, (0, 0), mask=overlay)
            # Note: save this image to the higher level directory
            img_pil.save(os.path.join(img_dir, f'{img_fname_stub}_combined_overlay_{args.model_type}.png'))

        t1 = datetime.now()
        print(f'Processed image in {(t1 - t0).total_seconds()} seconds')
