import types

import torch


def add_feature_extractor_method(model):
    # Add feature extractor method
    def my_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self.features_out = x.clone()
        x = self.output(x)
        return x

    def get_features(self):
        return self.features_out

    model.forward = types.MethodType(my_forward, model)
    model.get_features = types.MethodType(get_features, model)

    return model


def load_model(model, model_path: str):
    model.load_state_dict(torch.load(model_path))
    model = add_feature_extractor_method(model)
    model.eval()
    return model
