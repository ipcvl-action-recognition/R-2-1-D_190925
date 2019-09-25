import imgaug.augmenters as iaa
import cv2
import numpy as np
class ImageArg:
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8),
                order             = [0, 1], #interpolation
                cval              = 255
            ),
            iaa.Fliplr(0.5)
        ])

    # img_buffer의 값 c, l, h, w 순으로 나온다고 가정
    def ImageArgument(self, img_buffer):
        seq_det = self.seq._to_deterministic()
        buffer = img_buffer.transpose((1, 2, 3, 0))  # f, h, w, c

        buffer_aug = [seq_det.augment_image(frame) for frame in buffer]
        buffer_aug = np.array(buffer_aug)
        buffer_reshape = buffer_aug.transpose((3, 0, 1, 2))  # c, l, h, w
        return buffer_reshape