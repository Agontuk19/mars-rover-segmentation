"""Build a model from an experiment config dict."""
from src.models.unet import UNet


def build_model(cfg, num_classes=4):
    kind = cfg["model"]

    if kind == "unet_scratch":
        return UNet(in_channels=3, num_classes=num_classes, base=cfg.get("base", 32))

    if kind == "smp_unet":
        # imported lazily so notebooks without the package still work for other models
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=cfg.get("encoder_weights", "imagenet"),
            in_channels=3,
            classes=num_classes,
        )

    raise ValueError(f"Unknown model type: {kind}")