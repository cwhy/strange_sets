from strange_sets.count import CountSquares
import numpy as np
from pathlib import Path

dat = CountSquares(_dim_X=(28, 28), _dim_y=10, n_trials=30)
dat_mb = dat.sample(100)
out_dir = f'{Path.home()}/Data/strange/count'
np.save(f'{out_dir}/x0', dat_mb.x)
np.save(f'{out_dir}/y0', dat_mb.y)
