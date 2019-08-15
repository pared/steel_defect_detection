from src.utils import PROJECT_ROOT
import pandas as pd

def convert(csv):
    as_map = {}
    for index, row in csv.iterrows():
        id_class = row["ImageId_ClassId"]
        rle_mask_encoding = row["EncodedPixels"]

        image, class_id = id_class.split("_")
        class_id = int(class_id)

        if image not in as_map.keys():
            as_map[image] = {}

        as_map[image][class_id] = rle_mask_encoding

    new_data = [(image_id, rles[1], rles[2], rles[3], rles[4]) for
                image_id, rles in as_map.items()]
    del as_map

    return pd.DataFrame(new_data, columns=["image", "1", "2", "3", "4"])

csv_path = PROJECT_ROOT / "data" / "interim" / "train.csv"
csv = pd.read_csv(csv_path)

converted = convert(csv)
converted.to_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")