from strange_sets.count import CountSquares
from MLkit.mpl_helper import visualize_matrix

dat = CountSquares()
dat_mb = dat.sample(100)
p, _, _ = visualize_matrix(dat_mb.x[0, :, :])
print(dat_mb.x.shape)
print(dat_mb.y.shape)
p.show()
