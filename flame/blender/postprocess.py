import glob
import os
import random

from PIL import Image
from tqdm import tqdm

file_path_type = "../../data/backgrounds/*/*/*.jpg"


def remove_transparency(im, background, bg_colour=(255, 255, 255)):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        alpha = im.convert('RGBA').split()[-1]

        bg = Image.new("RGBA", im.size, bg_colour + (255,))

        bg.paste(background)
        bg.paste(im, mask=alpha)

        return bg

    else:
        return im


def main():
    # post process the rendered images.
    renders = glob.glob("D:/python_code/FacialReconstruction/out/renders/*.png")[26075 + 12072:]
    bgs = random.sample(glob.glob(file_path_type), len(renders))

    for (i, filename) in enumerate(tqdm(renders)):
        img = Image.open(filename)
        img = img.crop((512 // 2 - 200, 512 // 2 - 200, 512 // 2 + 200, 512 // 2 + 200))
        img = img.resize((256, 256), resample=Image.NEAREST)
        img = remove_transparency(img, Image.open(bgs[i]), bg_colour=(0, 0, 0))
        img.save(os.path.join("../../data/synthetic/renders/", filename.split("\\")[-1]))


if __name__ == "__main__":
    main()
