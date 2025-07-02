import numpy as np
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple
import random
from common.loggerConfig import logger
import json

def get_moving_avgs(arr, window, convolution_mode):
    flat = np.array(arr, dtype=float).flatten()
    kernel = np.ones(window, dtype=float)
    return np.convolve(flat, kernel, mode=convolution_mode) / window


def load_existing_model(model_path: Path) -> np.ndarray:
    model = None
    if model_path and model_path.exists():
        try:
            f = open(model_path, 'rb')
            model = pickle.load(f)
            f.close()
            logger.info(f"Model loaded from {model_path}")
            if isinstance(model, np.ndarray):
                logger.info(f"Model shape: {model.shape}")
        except Exception as e:
            logger.exception(f"Exception occured while loading {model_path}: {e}")
    elif model_path:
        logger.error(f"Provided path does not exists: {model_path}")
    return model

def load_existing_model_metadata(model_path: Path) -> Dict[str, Any]:
    """
    Load and return the metadata JSON for a given model file.

    The metadata is expected to live alongside the model file with a .json suffix.
    E.g., "foo.pt" → "foo.json".

    Raises:
        FileNotFoundError: if the .json metadata file does not exist.
        json.JSONDecodeError: if the file exists but is not valid JSON.
    """
    if model_path is None:
        return {}
    meta_data_path = model_path.with_suffix('.json')
    if not meta_data_path.exists():
        logger.warning(f"Metadata file not found: {meta_data_path}")
        return {}

    with meta_data_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {meta_data_path}: {metadata}")
    return metadata

def save_trained_model_metadata(model_path: Path, metadata: Dict):
    meta_data_path = model_path.with_suffix('.json')
    # ensure the directory exists
    meta_data_path.parent.mkdir(parents=True, exist_ok=True)
    # write out the JSON
    with meta_data_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    logger.info(f"Model metdata saved into {meta_data_path}: {metadata}")

def save_trained_model(model_path: Path, model: Any):

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(model_path,"wb")
        pickle.dump(model, f)
        f.close()
    except Exception as e:
        logger.exception(f"Exception occured while saving the file to {model_path}: {e}")

def split_dict(data: Dict, test_size: float) -> Tuple[Dict, Dict]:
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    items = list(data.items())
    random.shuffle(items)

    split_index = int(len(items) * (1 - test_size))
    train_items = items[:split_index]
    test_items = items[split_index:]

    train_data = dict(train_items)
    test_data = dict(test_items)

    return train_data, test_data