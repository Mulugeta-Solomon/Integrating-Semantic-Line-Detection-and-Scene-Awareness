from . import models

def build_model(cfg):
    model = models.Model(cfg)
    return model