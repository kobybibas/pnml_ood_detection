import logging

import torch

from lit_utils.baseline_lit_utils import LitBaseline

logger = logging.getLogger(__name__)


class LitOdin(LitBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 0.0
        self.temperature = 1.0

    def set_odin_params(self, eps=0.0, temperature=1.0):
        self.eps = eps
        self.temperature = temperature
        logger.info(f"set_odin_params: [eps temperature]=[{eps} {temperature}]")

    def forward(self, x):
        self.model.eval()

        # Fine-tune image
        if self.eps > 0.0:
            with torch.enable_grad():
                x_perturbs = self.perturb_input(
                    self.model, x, self.eps, self.temperature, self.model_name
                )
        else:
            x_perturbs = x

        logits, features, logits_w_norm_features = super().forward(x_perturbs)
        logits = logits / self.temperature
        # logits_w_norm_features = logits_w_norm_features / self.temperature
        return logits, features, logits_w_norm_features

    @staticmethod
    def perturb_input(
        model, images, epsilon: float, temperature: float, model_name: str
    ):
        """
        Execute adversarial attack on the input image.
        :param model: pytorch model to use.
        :param images: image to attack.
        :param epsilon: the attack strength
        :param temperature: smoothing factor of the logits.
        :param model_name: name of architecture
        :return: attacked image
        """
        criterion = torch.nn.CrossEntropyLoss()
        model.zero_grad()

        # Forward
        images.requires_grad = True
        outputs = model(images)

        # Using temperature scaling
        outputs = outputs / temperature

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        pseudo_labels = torch.argmax(outputs, dim=1).detach()
        loss = criterion(outputs, pseudo_labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        if model_name == "densenet":
            gradient.index_copy_(
                1,
                torch.LongTensor([0]).cuda(),
                gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([1]).cuda(),
                gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([2]).cuda(),
                gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0),
            )
        elif model_name == "resnet":
            gradient.index_copy_(
                1,
                torch.LongTensor([0]).cuda(),
                gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([1]).cuda(),
                gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994),
            )
            gradient.index_copy_(
                1,
                torch.LongTensor([2]).cuda(),
                gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010),
            )
        else:
            raise ValueError(f"{model_name} is not supported")

        # Adding small perturbations to images
        image_perturbs = torch.add(images.data, -gradient, alpha=epsilon)
        image_perturbs.requires_grad = False
        model.zero_grad()
        return image_perturbs
