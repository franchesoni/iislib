from typing import List, Union
import segmentation_models_pytorch as smp
import torch

# useful for all smp models
class EarlySMP(torch.nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, smp_model_class, smp_model_kwargs_dict, in_channels=6, classes=2):
        assert smp_model_kwargs_dict.has_key('encoder_weights'), 'Sorry, default behavior is undefined. To use pretrained weights pass "imagenet" as "encoder_weights" value.'
        smp_model_kwargs_dict['in_channels'] = in_channels
        smp_model_kwargs_dict['classes'] = classes
        self.model = smp_model_class(**smp_model_kwargs_dict)
        self.model_preprocessing_fn = smp.encoders.get_preprocessing_fn(smp_model_kwargs_dict['encoder_name']) if smp_model_kwargs_dict.get('encoder_weights') else None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.model_preprocessing_fn(x) if self.model_preprocessing_fn else x
        return self.model(x)

class EncodeSMP(torch.nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, smp_model_class, smp_aux_model_class, smp_model_kwargs_dict, smp_aux_model_kwargs_dict, smp_model_classes=2):
        assert smp_model_kwargs_dict.has_key('encoder_weights'), 'Sorry, default behavior is undefined. To use pretrained weights pass "imagenet" as "encoder_weights" value.'
        smp_model_kwargs_dict['classes'] = smp_model_classes
        self.model = smp_model_class(**smp_model_kwargs_dict)
        self.model_preprocessing_fn = smp.encoders.get_preprocessing_fn(smp_model_kwargs_dict['encoder_name']) if smp_model_kwargs_dict.get('encoder_weights') else None
        self.aux_model = smp_aux_model_class(**smp_aux_model_kwargs_dict)
        self.aux_model_preprocessing_fn = smp.encoders.get_preprocessing_fn(smp_aux_model_kwargs_dict['encoder_name']) if smp_aux_model_kwargs_dict.get('encoder_weights') else None

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> List[torch.Tensor]:
        x = self.model_preprocessing_fn(x) if self.model_preprocessing_fn else x
        encoded_x = self.model.encoder(x)
        aux = self.aux_model_preprocessing_fn(aux) if self.aux_model_preprocessing_fn else aux
        encoded_aux = self.aux_model.encoder(aux)
        fusion_code = [x_code + aux_code for x_code, aux_code in zip(encoded_x, encoded_aux)]
        return self.model.segmentation_head(self.model.decoder(fusion_code))

