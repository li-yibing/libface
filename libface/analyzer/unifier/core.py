import torch
from libface.base import BaseProcessor
from libface.datastruct import ImageData
from libface.logger import LoggerJsonFile
from torchvision import transforms

logger = LoggerJsonFile().logger


class FaceUnifier(BaseProcessor):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """FaceUnifier is a transform based processor that can unify sizes of all faces and normalize them between 0 and 1.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
        """
        super().__init__(transform, device, optimize_transform)

    def run(self, data: ImageData) -> ImageData:
        """Runs unifying transform on each face tensor one by one.

        Args:
            data (ImageData): ImageData object containing the face tensors.

        Returns:
            ImageData: ImageData object containing the unified face tensors normalized between 0 and 1.
        """
        for indx, face in enumerate(data.faces):
            data.faces[indx].tensor = self.transform(face.tensor)

        return data
