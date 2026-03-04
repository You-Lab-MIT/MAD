# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import random

from torchvision import transforms

from .transforms import GaussianBlur, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, make_normalize_transform


logger = logging.getLogger("dinov2")


def _get_normalize_params(mean=None, std=None):
    """Return (mean, std) for normalization. None means use ImageNet default."""
    mean = tuple(mean) if mean is not None else IMAGENET_DEFAULT_MEAN
    std = tuple(std) if std is not None else IMAGENET_DEFAULT_STD
    return mean, std


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        normalize_mean=None,
        normalize_std=None,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        mean, std = _get_normalize_params(normalize_mean, normalize_std)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class RandomMixedViewAugmentation(object):
    """
    Augmentation pipeline that randomly selects either the 'morphology' or 'neighborhood'
    image from the sample and applies the standard DINO augmentation (self-view).

    This allows training on a mixed distribution of both views without forcing
    an explicit pairing between them (which can be unstable).
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        morphology_key: str = "morphology",
        neighborhood_key: str = "neighborhood",
        normalize_mean=None,
        normalize_std=None,
    ):
        self.morphology_key = morphology_key
        self.neighborhood_key = neighborhood_key

        self.base_aug = DataAugmentationDINO(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    def __call__(self, sample_dict):
        if not isinstance(sample_dict, dict):
            return self.base_aug(sample_dict)

        if random.random() < 0.5:
            image = sample_dict.get(self.morphology_key)
            if image is None:
                raise KeyError(f"Sample dict missing '{self.morphology_key}'")
        else:
            image = sample_dict.get(self.neighborhood_key)
            if image is None:
                raise KeyError(f"Sample dict missing '{self.neighborhood_key}'")

        return self.base_aug(image)


class SingleViewAugmentation(object):
    """
    Apply the standard DINO augmentation pipeline to a single image fetched from a sample dict.

    Useful when a dataset returns multiple views (e.g. morphology & neighborhood) but we want
    both global and local crops to originate from only one of them.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        source_key: str = "morphology",
        normalize_mean=None,
        normalize_std=None,
    ):
        self.source_key = source_key
        self.base_aug = DataAugmentationDINO(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    def __call__(self, sample):
        if isinstance(sample, dict):
            image = sample.get(self.source_key)
            if image is None:
                raise KeyError(
                    f"Sample dict does not contain key '{self.source_key}'. Available keys: {list(sample.keys())}"
                )
        else:
            image = sample
        return self.base_aug(image)


class MorphologyNeighborhoodAugmentation(object):
    """
    Custom augmentation pipeline that aligns segmented-cell crops with their neighborhood context.

    - Global crops are sampled from the neighborhood image and are shared between the student and teacher.
    - Local (student-only) crops are sampled from the morphology image to emphasize the segmented cell.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        morphology_key: str = "morphology",
        neighborhood_key: str = "neighborhood",
        normalize_mean=None,
        normalize_std=None,
    ):
        self.morphology_key = morphology_key
        self.neighborhood_key = neighborhood_key
        self.local_crops_number = local_crops_number

        logger.info("###################################")
        logger.info("Using morphology/neighborhood data augmentation:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.neighborhood_geometric_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.morphology_geometric_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )
        local_transfo_extra = GaussianBlur(p=0.5)

        mean, std = _get_normalize_params(normalize_mean, normalize_std)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, sample_dict):
        if not isinstance(sample_dict, dict):
            raise TypeError("MorphologyNeighborhoodAugmentation expects a dict sample with morphology/neighborhood keys.")

        morphology_image = sample_dict.get(self.morphology_key)
        neighborhood_image = sample_dict.get(self.neighborhood_key)

        if morphology_image is None or neighborhood_image is None:
            raise KeyError(
                f"Sample dict must contain '{self.morphology_key}' and '{self.neighborhood_key}' PIL images. "
                f"Received keys: {list(sample_dict.keys())}"
            )

        output = {}

        im1_base = self.neighborhood_geometric_global(neighborhood_image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.neighborhood_geometric_global(neighborhood_image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = [
            self.local_transfo(self.morphology_geometric_local(morphology_image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class MorphologyNeighborhoodSameViewAugmentation(object):
    """
    Same-view only: student morph vs teacher morph, student micro vs teacher micro.
    Like standard DINO but with two independent streams (morphology and microenvironment).
    Outputs 4 global crops: [morph_1, morph_2, micro_1, micro_2], 1 local from morph.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        morphology_key: str = "morphology",
        neighborhood_key: str = "neighborhood",
        normalize_mean=None,
        normalize_std=None,
    ):
        self.morphology_key = morphology_key
        self.neighborhood_key = neighborhood_key

        mean, std = _get_normalize_params(normalize_mean, normalize_std)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ])
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        global_extra1 = GaussianBlur(p=1.0)
        global_extra2 = transforms.Compose([GaussianBlur(p=0.1), transforms.RandomSolarize(threshold=128, p=0.2)])
        local_extra = GaussianBlur(p=0.5)

        self.morph_global = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.micro_global = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.morph_local = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.global_transfo1 = transforms.Compose([color_jittering, global_extra1, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_extra2, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_extra, self.normalize])

        logger.info("Using MorphologyNeighborhoodSameViewAugmentation (morph-morph, micro-micro only)")

    def __call__(self, sample_dict):
        morphology_image = sample_dict[self.morphology_key]
        neighborhood_image = sample_dict[self.neighborhood_key]

        m1 = self.global_transfo1(self.morph_global(morphology_image))
        m2 = self.global_transfo2(self.morph_global(morphology_image))
        n1 = self.global_transfo1(self.micro_global(neighborhood_image))
        n2 = self.global_transfo2(self.micro_global(neighborhood_image))

        return {
            "global_crops": [m1, m2, n1, n2],
            "global_crops_teacher": [m1, m2, n1, n2],
            "local_crops": [self.local_transfo(self.morph_local(morphology_image))],
            "offsets": (),
        }


class MorphologyNeighborhoodFourWayAugmentation(object):
    """
    Four-way cross: all pairs among {student morph, teacher morph, student micro, teacher micro}.
    Outputs 4 global crops: [morph_1, morph_2, micro_1, micro_2], 1 local from morph.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        morphology_key: str = "morphology",
        neighborhood_key: str = "neighborhood",
        normalize_mean=None,
        normalize_std=None,
    ):
        self.morphology_key = morphology_key
        self.neighborhood_key = neighborhood_key

        mean, std = _get_normalize_params(normalize_mean, normalize_std)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ])
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        global_extra1 = GaussianBlur(p=1.0)
        global_extra2 = transforms.Compose([GaussianBlur(p=0.1), transforms.RandomSolarize(threshold=128, p=0.2)])
        local_extra = GaussianBlur(p=0.5)

        self.morph_global = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.micro_global = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.morph_local = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.global_transfo1 = transforms.Compose([color_jittering, global_extra1, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_extra2, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_extra, self.normalize])

        logger.info("Using MorphologyNeighborhoodFourWayAugmentation (all pairs)")

    def __call__(self, sample_dict):
        morphology_image = sample_dict[self.morphology_key]
        neighborhood_image = sample_dict[self.neighborhood_key]

        m1 = self.global_transfo1(self.morph_global(morphology_image))
        m2 = self.global_transfo2(self.morph_global(morphology_image))
        n1 = self.global_transfo1(self.micro_global(neighborhood_image))
        n2 = self.global_transfo2(self.micro_global(neighborhood_image))

        return {
            "global_crops": [m1, m2, n1, n2],
            "global_crops_teacher": [m1, m2, n1, n2],
            "local_crops": [self.local_transfo(self.morph_local(morphology_image))],
            "offsets": (),
        }


class RandomMixedViewAugmentation(object):
    """
    Augmentation pipeline that randomly selects either the 'morphology' or 'neighborhood'
    image from the sample and applies the standard DINO augmentation (self-view).

    This allows training on a mixed distribution of both views without forcing
    an explicit pairing between them (which can be unstable).
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        morphology_key: str = "morphology",
        neighborhood_key: str = "neighborhood",
        normalize_mean=None,
        normalize_std=None,
    ):
        self.morphology_key = morphology_key
        self.neighborhood_key = neighborhood_key

        self.base_aug = DataAugmentationDINO(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

    def __call__(self, sample_dict):
        if not isinstance(sample_dict, dict):
            return self.base_aug(sample_dict)

        if random.random() < 0.5:
            image = sample_dict.get(self.morphology_key)
            if image is None:
                raise KeyError(f"Sample dict missing '{self.morphology_key}'")
        else:
            image = sample_dict.get(self.neighborhood_key)
            if image is None:
                raise KeyError(f"Sample dict missing '{self.neighborhood_key}'")

        return self.base_aug(image)
