"""Módulo de entrenamiento"""
from .train import train_epoch, validate
from .inference import predict_on_video

__all__ = ["train_epoch", "validate", "predict_on_video"]
