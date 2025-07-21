import logging
import os
import shutil
from pathlib import Path
from typing import Union, Optional, Dict
from urllib import request

import torch
import yaml
from numpy import ndarray
from torch._appdirs import user_cache_dir
from yaml import YAMLError

from plants_sm.io.yaml import read_yaml

logger = logging.getLogger(__name__)

_module_dir: Path = Path(os.path.dirname(os.path.abspath(__file__)))


def get_device(device: Union[None, str, torch.device] = None, cuda_as_default=False) -> torch.device:
    """
    Returns what the user specified, or defaults to the GPU,
    with a fallback to CPU if no GPU is available.

    Parameters
    ----------
    device: Union[None, str, torch.device]
        The device to use. If None, defaults to GPU, with fallback to CPU.
    cuda_as_default: bool
        Whether to use CUDA as default device.
    Returns
    -------
    torch.device
        The device to use.
    """
    if isinstance(device, torch.device):
        return device
    elif device is None:
        return torch.device("cpu")
    elif "cuda" in device:
        is_available = torch.cuda.is_available()
        if is_available:
            return torch.device(device)
        else:
            logger.warning("CUDA is not available, falling back to CPU.")
            return torch.device("cpu")
    elif torch.cuda.is_available() and cuda_as_default:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:
    """
    Read config from path to file.

    Parameters
    ----------
    config_path: Union[str, Path]
        Path to config file.
    preserve_order: bool
        Whether to preserve order of config file.

    Returns
    -------
    result of parsing the YAML file: dict
    """
    read_yaml(config_path, preserve_order=preserve_order)
    with open(config_path, "r") as fp:
        try:
            if preserve_order:
                return yaml.load(fp, Loader=yaml.Loader)
            else:
                return yaml.safe_load(fp)
        except YAMLError as e:
            raise YAMLError(
                f"Could not parse configuration file at '{config_path}' as yaml. "
                "Formatting mistake in config file? "
                "See Error above for details."
            ) from e


def get_model_file(model: Optional[str] = None,
                   file: Optional[str] = None,
                   overwrite_cache: bool = False) -> str:
    """
    If the specified asset for the model is in the user cache, returns the
    location, otherwise downloads the file to cache and returns the location.

    Parameters
    ----------
    model: Optional[str]
        The model to get the file for. If None, returns the file for the
        default model.
    file: Optional[str]
        The file to get. If None, returns the default file for the model.
    overwrite_cache: bool
        Whether to overwrite the cache if the file is already present.
    """
    cache_path = Path(user_cache_dir("plants_sm")).joinpath(model).joinpath(file)
    if not overwrite_cache and cache_path.is_file():
        logger.info(f"Loading {file} for {model} from cache at '{cache_path}'")
        return str(cache_path)

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    _defaults: Dict[str, Dict[str, str]] = read_config_file(_module_dir / "defaults.yml")
    url = _defaults.get(model, {}).get(file)

    # Since the files are not user provided, this must never happen
    assert url, f"File {file} for {model} doesn't exist."

    logger.info(f"Downloading {file} for {model} and storing it in '{cache_path}'")

    req = request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    })

    with request.urlopen(req) as response, open(cache_path, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)

    return str(cache_path)


def reduce_per_protein(embedding: ndarray) -> ndarray:
    """
    Reduce the embedding of a protein to a single vector.

    Parameters
    ----------
    embedding: ndarray
        The embedding of a protein.

    Returns
    -------
    ndarray
        The reduced embedding.
    """
    # This is `h_avg` in jax-unirep terminology
    return embedding.mean(axis=0)
