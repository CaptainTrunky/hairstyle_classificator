import torch as T

from torchvision.models import shufflenet_v2_x0_5 as Backbone


class HairStyleClassifier(T.nn.Module):
    def __init__(self):
        super().__init__()

        self._backbone = Backbone(pretrained=True, progress=True)

        for param in self._backbone.parameters():
            param.requires_grad = False

        num_classes = 1

        channels_in = self._backbone._stage_out_channels[-1]

        self._backbone.fc = T.nn.Linear(channels_in, num_classes)

    def forward(self, x):
        return self._backbone(x)


def get_model():
    return HairStyleClassifier()


