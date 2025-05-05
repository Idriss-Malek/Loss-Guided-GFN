
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.gflownet.tbgflownetV2 import TBGFlowNetV2

__all__ = [
    "TBGFlowNetV2"
]