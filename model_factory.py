"""Python file to instantite the model and the transform that goes with it."""
from model import Deit
from data import train_transforms, val_transforms


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform_train()
        self.transform2 = self.init_transform_val()

    def init_model(self):
        if self.model_name == "deit_base_distilled_patch16_384":
            return Deit()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform_train(self):
        if self.model_name == "deit_base_distilled_patch16_384":
            return train_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def init_transform_val(self):
        if self.model_name == "deit_base_distilled_patch16_384":
            return val_transforms
        else:
            raise NotImplementedError("Transform not implemented")


    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_transform2(self):
        return self.transform2

    def get_all(self):
        return self.model, self.transform, self.transform2

