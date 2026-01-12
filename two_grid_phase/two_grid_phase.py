from fealpy.backend import backend_manager as bm
from fealpy.cfd import TwoGridModel

model = TwoGridModel(fine_model, coarsen_model)
