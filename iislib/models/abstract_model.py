from abc import ABC
from abc import abstractmethod

import torch
from clicking.robots import Clicks


class IISBaseModel(ABC):
    @abstractmethod
    def forward(
        self, x: torch.Tensor, z: dict, pcs: Clicks, ncs: Clicks
    ) -> tuple[torch.Tensor, dict]:
        """Inference step

        Args:
            x (torch.Tensor): input image, (B, 3, H, W)
            z (dict): auxiliary inputs (e.g. border or prev_output)
            pcs (Clicks): positive clicks, indexed by (interaction, batch_element, click_n)
            ncs (Clicks): negative clicks, indexed by (interaction, batch_element, click_n)

        Returns:
            torch.Tensor: predicted mask `y` (can be non binary), (B, 1, H, W)
            dict: next state `z`
        """

    @abstractmethod
    def init_z(self, image, target):
        """Initialization of state

        Args:
            image (torch.Tensor): input image, (B, 3, H, W) float [0, 1]
            target (torch.Tensor): target mask, (B, 1, H, W) binary {0, 1}

        Returns:
            dict: first state `z`
        """

    @abstractmethod
    def init_y(self, image, target):
        """Initialization of prediction

        Args:
            image (torch.Tensor): input image, (B, 3, H, W) float [0, 1]
            target (torch.Tensor): target mask, (B, 1, H, W) binary {0, 1}

        Returns:
            torch.Tensor: first prediction `y`, (B, 1, H, W), float [0, 1]
        """
