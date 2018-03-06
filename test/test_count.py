from strange_sets.count import CountSquares
from MLkit.mpl_helper import visualize_matrix

dat = CountSquares(_dim_X=(28, 28), _dim_y=10)
dat_mb = dat.sample(100)
p, _, _ = visualize_matrix(dat_mb.x[0, :, :], dpi=150)
print(dat_mb.x.shape)
print(dat_mb.y)
p.show()
