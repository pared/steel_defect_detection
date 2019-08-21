from fastai.vision import *
from src.utils import PROJECT_ROOT

class DefectsSegList(SegmentationLabelList):
    def open(self, id_rles):
        image_id, rles = id_rles[0], id_rles[1:]
        shape = open_image(self.path/image_id).shape[-2:]
        final_mask = torch.zeros((1, *shape))
        for k, rle in enumerate(rles):
            if isinstance(rle, str):
                mask = open_mask_rle(rle, shape).px.permute(0, 2, 1)
                final_mask += (k+1)*mask
        return ImageSegment(final_mask)

def load_data(bs=32, size=(256, 1600)):
    labels_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    images_path = PROJECT_ROOT / "data" / "interim" / "train_images"
    train_list = (SegmentationItemList.
                  from_csv(images_path, labels_path).
                  split_by_rand_pct(valid_pct=0.2).
                  label_from_df(cols=list(range(5)), label_cls=DefectsSegList, classes=[]).
                  transform(size=size, tfm_y=True).
                  databunch(bs=bs, num_workers=0).
                  normalize(imagenet_stats))
    return train_list
