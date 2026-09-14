import typer
import os
import math
from PIL import Image, ImageDraw
from glob import glob
from brepdgt.utils.vis import concat_h_pil, concat_v_pil
from typing import List
import PIL

app = typer.Typer()


def get_content_bbox(img):
    # Convert to RGBA if not already
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Get non-white pixels (threshold at 250)
    data = img.getdata()
    non_white = [
        (x, y)
        for y in range(img.height)
        for x in range(img.width)
        if any(v < 240 for v in data[y * img.width + x][:3])
    ]

    if not non_white:
        return None

    # Get bounding box
    left = min(x for x, y in non_white)
    top = min(y for x, y in non_white)
    right = max(x for x, y in non_white)
    bottom = max(y for x, y in non_white)

    return (left, top, right, bottom)


def crop_with_margin(img):
    original_size = img.size
    bbox = get_content_bbox(img)
    if not bbox:
        return img

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    # Make it square based on the larger dimension
    size = max(width, height)

    # Add 10% margin
    margin = int(size * 0.1)

    # Calculate new bounds keeping aspect ratio 1:1 and centering the content
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    half_size = (size + 2 * margin) // 2

    new_left = max(0, center_x - half_size)
    new_right = min(img.width, center_x + half_size)
    new_top = max(0, center_y - half_size)
    new_bottom = min(img.height, center_y + half_size)

    # Crop and resize back to original size
    cropped = img.crop((new_left, new_top, new_right, new_bottom))
    return cropped.resize(original_size, Image.Resampling.LANCZOS)


def cat_imgs_with_paths(
    img_paths: List[str],
    img_out_path: str,
    n_imgs: int = 100,
    prefix: str = "",
    postfix: str = "",
    add_name: bool = True,
    crop: bool = False,
    chunk: bool = False,
):
    if chunk:
        # For chunked rendering: 8 rows x 5 columns = 40 images per chunk
        rows, cols = 8, 5
        assert (
            n_imgs % 40 == 0
        ), f"n_imgs must be divisible by 40 when using chunks, got {n_imgs}"
        n_chunks = n_imgs // 40
        base, ext = os.path.splitext(img_out_path)

        # Process each chunk
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * 40
            chunk_img_paths = img_paths[start_idx : start_idx + 40]
            chunk_out_path = f"{base}_{chunk_idx}{ext}"

            # Process single chunk
            imgs, img_cnt = [], 0
            for h in range(rows):
                imgs_h = []
                for w in range(cols):
                    img = PIL.Image.open(chunk_img_paths[img_cnt])
                    if crop:
                        img = crop_with_margin(img)
                    if add_name:
                        name = os.path.basename(chunk_img_paths[img_cnt]).replace(
                            ".png", ""
                        )
                        d = ImageDraw.Draw(img)
                        d.text(
                            (10, 10),
                            f"{name}",
                            fill=(0, 0, 0, 255),
                        )
                    imgs_h.append(img)
                    img_cnt += 1
                img_h_concat = concat_h_pil(imgs_h)
                imgs.append(img_h_concat)
            imgs = concat_v_pil(imgs)
            imgs.save(chunk_out_path)
    else:
        # Original square layout
        rows = cols = int(math.sqrt(n_imgs))
        assert (
            n_imgs == rows**2
        ), f"n_imgs must be square number when not using chunks, got {n_imgs}"

        imgs, img_cnt = [], 0
        for h in range(rows):
            imgs_h = []
            for w in range(cols):
                img = PIL.Image.open(img_paths[img_cnt])
                if crop:
                    img = crop_with_margin(img)
                if add_name:
                    name = os.path.basename(img_paths[img_cnt]).replace(".png", "")
                    d = ImageDraw.Draw(img)
                    d.text(
                        (10, 10),
                        f"{name}",
                        fill=(0, 0, 0, 255),
                    )
                imgs_h.append(img)
                img_cnt += 1
            img_h_concat = concat_h_pil(imgs_h)
            imgs.append(img_h_concat)
        imgs = concat_v_pil(imgs)
        imgs.save(img_out_path)


@app.command()
def main(
    img_dir: str,
    n_imgs: int = 100,
    prefix: str = "",
    postfix: str = "",
    img_out_path: str = None,
    avoid_key: str = None,
    add_name: bool = False,
    crop: bool = False,
    chunk: bool = False,
):
    img_paths = glob(os.path.join(img_dir, f"{prefix}*{postfix}.png"))
    if avoid_key:
        img_paths = [img_path for img_path in img_paths if avoid_key not in img_path]
    img_paths = list(sorted(img_paths))[:n_imgs]

    if img_out_path is None:
        # export cat image under the default render directory
        img_out_name = f"first_{n_imgs}_samples"
        if prefix != "":
            img_out_name = f"{prefix}_" + img_out_name
        if postfix != "":
            img_out_name = img_out_name + f"_{postfix}"
        img_out_path = os.path.join(img_dir, f"{img_out_name}.png")

    cat_imgs_with_paths(
        img_paths, img_out_path, n_imgs, prefix, postfix, add_name, crop, chunk
    )


@app.command()
def dummy():
    pass


if __name__ == "__main__":
    app()
