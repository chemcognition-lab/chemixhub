from jaxtyping import Bool, Float
from torch import Tensor

# Basic aliases
Tensor = Tensor
# Specific types
ManyMixTensor = Float[Tensor, "b mol emb many"]
MixTensor = Float[Tensor, "batch mol emb"]
MaskTensor = Bool[Tensor, "batch mol"]
EmbTensor = Float[Tensor, "batch emb"]
PredictionTensor = Float[Tensor, "batch out"]
ManyEmbTensor = Float[Tensor, "batch emb many"]
