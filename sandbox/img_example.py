import matplotlib.pyplot as pl
import fisheye

data = pl.imread('./example.png')

R = 300
focus = [400.,400.]

F = fisheye.fisheye(300,xw=0.3,mode='default',d=2)
F.set_focus(focus)

transformed = fisheye.apply_to_image(data, F)

fig, ax = pl.subplots(1,2,figsize=(8,3))

ax[0].imshow(data)
ax[1].imshow(transformed)

ax[0].axis('off')
ax[1].axis('off')

pl.show()

