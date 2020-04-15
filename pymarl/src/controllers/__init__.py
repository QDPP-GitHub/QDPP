REGISTRY = {}

from .basic_controller import BasicMAC
from .noise_controller import NoiseMAC
from .qdpp_controller import QDPPMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["noise_mac"] = NoiseMAC
REGISTRY["qdpp_mac"] = QDPPMAC