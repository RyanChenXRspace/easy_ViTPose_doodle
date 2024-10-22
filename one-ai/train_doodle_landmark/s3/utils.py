import os
from PIL import Image
from PIL import ExifTags
from pillow_heif import register_heif_opener


register_heif_opener()


def convert_image_to_png(file_path):
    try:
        img = Image.open(file_path)

        longest_side = max(img.size)
        scale_factor = 1024 / longest_side
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}.png"
        img.save(output_path, "PNG")
        print(f"save PNG: {output_path}")

    except Exception as e:
        print(f": {e}")


if __name__ == "__main__":
    convert_image_to_png('452b0db5-9615-46ac-a564-60f8ff2e1cb6')
