"""P5.5 spatial-temporal encoder (plain PyTorch): heterogeneous GATv2 per history frame + causal dilated TCN per agent.

History-only inputs, image-space features, no decoder / trajectory head / training. See ``encoder.py`` for the input and
output contracts, ``spatial.py`` for the attention equations and ``temporal.py`` for the causal design.
"""

from .config import EncoderConfig
from .encoder import EdgeBatch, EncoderInput, EncoderOutput, SpatioTemporalEncoder
from .scaling import FeatureScaler
from .spatial import HeteroGATv2Encoder
from .temporal import CausalDilatedTCN

__all__ = ["EncoderConfig", "EncoderInput", "EncoderOutput", "EdgeBatch", "SpatioTemporalEncoder", "FeatureScaler", "HeteroGATv2Encoder", "CausalDilatedTCN"]
