import numpy as np
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple
import random
from common.loggerConfig import logger

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
            logger.exception(f"Exception occured while loading {model_path}: {e}")
    elif model_path:
        logger.error(f"Provided path does not exists: {model_path}")
    return model

def save_trained_model(model_path: Path, model: Any):

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(model_path,"wb")
        pickle.dump(model, f)
        f.close()
    except Exception as e:
        logger.exception(f"Exception occured while saving the file to {model_path}: {e}")

def split_dict(data: Dict, test_size: float) -> Tuple[Dict, Dict]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    items = list(data.items())
    random.shuffle(items)

    split_index = int(len(items) * (1 - test_size))
    train_items = items[:split_index]
    test_items = items[split_index:]

    train_data = dict(train_items)
    test_data = dict(test_items)

    return train_data, test_data