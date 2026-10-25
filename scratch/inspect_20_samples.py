"""Inspect why overfit failed on some samples."""
import torch
import cv2
import pandas as pd
from src.training.dataset import IIIT5KDataset, DEFAULT_CONVERTER, ctc_collate_fn
from src.training.model import build_crnn

df = pd.read_csv("traindata.csv").head(20)
for idx, row in df.iterrows():
    print(f"[{idx:2d}] Img: {row['ImgName']} | GT: '{row['GroundTruth']}' | Encoded: {DEFAULT_CONVERTER.encode(row['GroundTruth'])}")
