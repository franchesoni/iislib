import torch

from models.custom.ritm.isegm.inference import utils


def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

def predict(model, image, prev_mask, pcs, ncs):
    assert len(pcs[0]) == len(ncs[0]), "batch size should be consistent"
    for n_interaction in range(len(ncs)):
        for n_element in range(len(ncs[0])):
            maxlen = max(
                len(pcs[n_interaction][n_element]), len(ncs[n_interaction][n_element])
            )
            pcs[n_interaction][n_element] = pcs[n_interaction][n_element] + [-1] * (
                maxlen - len(pcs[n_interaction][n_element])
            )
            ncs[n_interaction][n_element] = ncs[n_interaction][n_element] + [-1] * (
                maxlen - len(ncs[n_interaction][n_element])
            )

    points = []
    for n_element in range(len(pcs[0])):  # for element in batch
        points.append([])
        for n_interaction in range(len(pcs)):  # for each interaction
            for click in pcs[n_interaction][n_element]:
                if click == -1:
                    points[-1].append([-1, -1, -1])  # add all positive clicks
                else:
                    points[-1].append(
                        [click[0], click[1], n_interaction]
                    )  # add all positive clicks
        for n_interaction in range(len(ncs)):
            for click in ncs[n_interaction][n_element]:
                if click == -1:
                    points[-1].append([-1, -1, -1])  # add all positive clicks
                else:
                    points[-1].append(
                        [click[0], click[1], n_interaction]
                    )  # add all negative clicks

    input_image = torch.cat((image, prev_mask), dim=1)
    pred_logits = model.forward(input_image, torch.Tensor(points))['instances']
    prediction = torch.nn.functional.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                size=input_image.size()[2:])
    prediction = torch.sigmoid(prediction)
    return prediction, {'prev_prediction':prediction}


def initialize_z(image, target):
    return {'prev_prediction':torch.zeros_like(image[:, :1, :, :])}


def initialize_y(image, target):
    y = torch.zeros_like(target)
    return y


# define model here
checkpoint = (
    "/home/franchesoni/iis/iis_framework/models/custom/ritm/sbd_h18_itermask.pth"
)
model = utils.load_is_model(checkpoint, device="cpu")


def ritm(x, z, pcs, ncs, model=model):
    y, z = predict(model, x, z['prev_prediction'], pcs, ncs)
    return y, z
