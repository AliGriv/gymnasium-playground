import numpy as np
from pathlib import Path
import pickle
from typing import Any

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def load_existing_model(model_path: Path) -> np.ndarray:
    model = None
    if model_path and model_path.exists():
        try:
            f = open(model_path, 'rb')
            model = pickle.load(f)
            f.close()
        except Exception as e:
            print(f"Exception occured while loading {model_path}: {e}")
    elif model_path:
        print(f"Provided path does not exists: {model_path}")
    return model

def save_trained_model(model_path: Path, model: Any):

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(model_path,"wb")
        pickle.dump(model, f)
        f.close()
    except Exception as e:
        print(f"Exception occured while saving the file to {model_path}: {e}")