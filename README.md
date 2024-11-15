# RGB Road Scene Material Segmentation for the Urban-M4 project

This is a repackaged version of the original [RGB Road Scene Material Segmentation](https://github.com/kyotovision-public/RGB-Road-Scene-Material-Segmentation/?tab=readme-ov-file) repository. Please refer to the original [README](https://github.com/kyotovision-public/RGB-Road-Scene-Material-Segmentation/blob/main/README.md) as well.


# Installation
The code is in need of refactoring, so some effort has been made to make it easier to install. After cloning the repository, run the following to install the dependencies and the codebase as a local editable package:

```bash
$> pip install -e .
```

# Data

Unpack the archive containing images and pretrained weights into the root directory of the repository:

```bash
$> tar -xvzf data.tar.gz
```

# Notebooks

- [segmentation.ipynb](./notebooks/segmentation.ipynb): an example of a generic segmentation pipeline using a pretrained SegFormer model from Huggingface. It does not identify materials, only objects of the same class.
- [inference.ipynb](./notebooks/inference.ipynb) contains an example of running the RMSNet pipeline. It should output material classes.