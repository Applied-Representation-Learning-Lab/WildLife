from lightly.transforms import VICRegTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torchvision import transforms

def get_transform(config):
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((config.image_dim, config.image_dim)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ]
    )

    if config.dataset == "swedish":
        TRAIN_TRANSFORM = VICRegTransform(input_size=config.image_dim)

    return TRAIN_TRANSFORM, TEST_TRANSFORM
