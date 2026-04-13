
import os

from .utils.py_functional import is_package_available


if is_package_available("modelscope"):
    from modelscope.utils.hf_util import patch_hub


__version__ = "0.3.3.dev0"


if os.getenv("USE_MODELSCOPE_HUB", "0").lower() in ["true", "y", "1"]:
    if not is_package_available("modelscope"):
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope`.")

    patch_hub()
