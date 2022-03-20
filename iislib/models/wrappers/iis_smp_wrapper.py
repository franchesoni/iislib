from typing import List

import segmentation_models_pytorch as smp
import torch
from models.abstract_model import IISBaseModel


class EarlySMP(IISBaseModel, torch.nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(
        self,
        smp_model_class,
        smp_model_kwargs_dict,
        click_encoder,
        in_channels=6,
        classes=1,
    ):
        """`smp_model_kwargs_dict` should be a dict with at least 'encoder_weights' as a key. It is used to init the `smp_model_class`"""
        super().__init__()  # this should work because only torch.nn.Module has __init__
        assert (
            "encoder_weights" in smp_model_kwargs_dict
        ), 'Sorry, default behavior is undefined. To use pretrained weights\
        pass "imagenet" as "encoder_weights" value.'
        self.click_encoder = click_encoder
        smp_model_kwargs_dict["in_channels"] = in_channels
        smp_model_kwargs_dict["classes"] = classes
        self.model = smp_model_class(
            **smp_model_kwargs_dict
        ).float()  # does this work with lightning?
        self.model_preprocessing_fn = (
            smp.encoders.get_preprocessing_fn(
                smp_model_kwargs_dict["encoder_name"]
            )
            if smp_model_kwargs_dict.get("encoder_weights")
            else None
        )

    def forward(self, x, z, pcs, ncs):
        pos_encoding, neg_encoding = torch.zeros_like(
            x[:, :1, :, :]
        ), torch.zeros_like(x[:, :1, :, :])
        pos_encoding, neg_encoding = self.click_encoder(
            pcs, ncs, pos_encoding, neg_encoding, radius=5
        )
        aux = torch.concat(
            (z["prev_output"], pos_encoding, neg_encoding), axis=1
        )
        y = torch.sigmoid(self.x_aux_forward(x, aux))
        z["prev_output"] = y
        return y, z

    def x_aux_forward(
        self, x: torch.Tensor, aux: torch.Tensor
    ) -> torch.Tensor:
        x = (
            self.model_preprocessing_fn(x.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
            if self.model_preprocessing_fn
            else x
        )
        x_in = torch.cat(
            (x, aux), dim=1
        ).float()  # does this work with lightning?
        return self.model(x_in)

    def init_z(self, image, target):
        return {"prev_output": torch.zeros_like(target)}

    def init_y(self, image, target):
        return torch.zeros_like(target)


class EncodeSMP(torch.nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(
        self,
        click_encoder,
        smp_model_class,
        smp_aux_model_class,
        smp_model_kwargs_dict,
        smp_aux_model_kwargs_dict,
        smp_model_classes=1,
    ):
        super().__init__()
        assert (
            "encoder_weights" in smp_model_kwargs_dict
        ), 'Sorry, default\
             behavior is undefined. To use pretrained weights pass\
             "imagenet" as "encoder_weights" value.'
        self.click_encoder = click_encoder
        smp_model_kwargs_dict["classes"] = smp_model_classes
        self.model = smp_model_class(**smp_model_kwargs_dict)
        self.model_preprocessing_fn = (
            smp.encoders.get_preprocessing_fn(
                smp_model_kwargs_dict["encoder_name"]
            )
            if smp_model_kwargs_dict.get("encoder_weights")
            else None
        )
        self.aux_model = smp_aux_model_class(**smp_aux_model_kwargs_dict)
        self.aux_model_preprocessing_fn = (
            smp.encoders.get_preprocessing_fn(
                smp_aux_model_kwargs_dict["encoder_name"]
            )
            if smp_aux_model_kwargs_dict.get("encoder_weights")
            else None
        )

    def forward(self, x, z, pcs, ncs):
        raise NotImplementedError

    def x_aux_forward(
        self, x: torch.Tensor, aux: torch.Tensor
    ) -> List[torch.Tensor]:
        x = (
            self.model_preprocessing_fn(x.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
            if self.model_preprocessing_fn
            else x
        )
        encoded_x = self.model.encoder(x)
        aux = (
            self.aux_model_preprocessing_fn(aux.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
            if self.aux_model_preprocessing_fn
            else aux
        )
        encoded_aux = self.aux_model.encoder(aux)
        fusion_code = [
            x_code + aux_code
            for x_code, aux_code in zip(encoded_x, encoded_aux)
        ]
        return self.model.segmentation_head(self.model.decoder(*fusion_code))
