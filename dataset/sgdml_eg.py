import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io

r,_ = io.read_xyz('geometries/ethanol.xyz') # 9 atoms
print(r.shape) # (1,27)

model = np.load('models/ethanol.npz')
gdml = GDMLPredict(model)
e,f = gdml.predict(r)
print(e.shape) # (1,)
print(f.shape) # (1,27)
