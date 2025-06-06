import json
import gzip
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union, Optional
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.envs.toy_text.frozen_lake import is_valid
from common.loggerConfig import logger

class FrozenLakeMaps:
    """
    A utility class for generating, saving, and loading FrozenLake map datasets.
    """

    @staticmethod
    def generate_random_map(size: int = 8, seed: Optional[int] = None) -> List[str]:
        """
        Generate a single random valid FrozenLake map of given size x size.

        Args:
            size: The size of the square map to generate
            seed: optional seed to ensure the generation of reproducible maps
        Returns:
            A list of strings representing the map
        """
        try:
            return generate_random_map(size=size, seed=seed)
        except Exception as e:
            raise ValueError(f"Failed to generate map of size {size}: {str(e)}")

    @staticmethod
    def save_maps(maps_dict: Dict[str, List[str]],
                  filepath: Union[str, Path],
                  compress: bool = False,
                  chunk_size: int = 1000) -> None:
        """
        Save a dictionary of maps to a JSON file (optionally compressed).

        Args:
            maps_dict: Dictionary of maps where keys are map IDs and values are map grids
            filepath: Path to save the file
            compress: Whether to compress the output file using gzip
            chunk_size: Number of maps per chunk file
        """
        # Validate maps_dict structure
        for map_id, map_grid in maps_dict.items():
            if not isinstance(map_id, str):
                raise TypeError(f"Map ID must be a string, got {type(map_id)}")
            if not isinstance(map_grid, list) or not all(isinstance(row, str) for row in map_grid):
                raise TypeError(f"Map grid must be a list of strings for map ID {map_id}")

        filepath = Path(filepath)
        base = filepath.stem
        ext = ".json.gz" if compress else ".json"
        out_dir = filepath.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Chunk and save
        items = list(maps_dict.items())
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            chunk_filename = out_dir / f"{base}.{i // chunk_size}{ext}"
            try:
                if compress:
                    with gzip.open(chunk_filename, "wt", encoding="utf-8") as f:
                        json.dump(chunk, f)
                else:
                    with open(chunk_filename, "w", encoding="utf-8") as f:
                        json.dump(chunk, f)
            except Exception as e:
                raise IOError(f"Failed to save chunk to {chunk_filename}: {str(e)}")

    @staticmethod
    def load_maps(filepath: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Load a dictionary of maps from a JSON or compressed JSON file.

        Args:
            filepath: Path to the JSON file (may be gzip compressed if ending with .gz).
                      Can be path to directory containing multiple files.

        Returns:
            Dictionary of maps where keys are map IDs and values are map grids
        """
        def _load_single_file(file_path: Path) -> Dict[str, List[str]]:
            try:
                if str(file_path).endswith(".gz"):
                    with gzip.open(file_path, "rt", encoding="utf-8") as f:
                        maps_dict = json.load(f)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        maps_dict = json.load(f)
                return maps_dict
            except Exception as e:
                raise IOError(f"Failed to load maps from {file_path}: {str(e)}")
        filepath = Path(filepath)
        if filepath.is_dir():
            maps_dict = {}
            for file in filepath.iterdir():
                loaded_data = _load_single_file(file)
                if loaded_data:
                    maps_dict.update(loaded_data)
        else:
            maps_dict = _load_single_file(filepath)

        return maps_dict



    @classmethod
    def generate_dataset(cls, num_maps: int, size: int, filepath: Union[str, Path],
                     compress: bool = False, seed: Optional[int] = None,
                     max_attempts: int = 10000000) -> Dict[str, List[str]]:
        """
        Generate a dataset of unique valid random maps and save it to a file.

        Args:
            num_maps: Number of valid maps to generate
            size: Size of each map (size x size)
            filepath: Path to save the dataset
            compress: Whether to compress the output file
            seed: Optional random seed for reproducibility
            max_attempts: Maximum number of attempts to generate requested number of maps

        Returns:
            The generated maps dictionary containing only valid maps
        """
        if seed is not None:
            import random
            random.seed(seed)



        maps = {}
        unique_maps = set()  # To track unique maps and avoid duplicates
        attempts = 0

        with tqdm(total=num_maps, desc="Generating valid maps") as pbar:
            while len(maps) < num_maps and attempts < max_attempts:
                attempts += 1

                try:
                    # Generate a random map
                    map_grid = cls.generate_random_map(size=size)

                    # Convert to a tuple for hashing (to check uniqueness)
                    map_tuple = tuple(map_grid)
                    if map_tuple in unique_maps:
                        continue # don't waste time here
                    # Check if it's valid and not a duplicate
                    if is_valid(map_grid, size):
                        map_id = str(len(maps))
                        maps[map_id] = map_grid
                        unique_maps.add(map_tuple)
                        pbar.update(1)

                except Exception as e:
                    logger.warning(f"Failed during map generation: {str(e)}")
                    continue

        if len(maps) < num_maps:
            logger.warning(f"=Could only generate {len(maps)} valid unique maps out of requested {num_maps} after {attempts} attempts")

        cls.save_maps(maps, filepath, compress=compress)
        return maps