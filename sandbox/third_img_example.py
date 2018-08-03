import matplotlib.pyplot as pl
import fisheye


R = 300
focus = [400,400]

F = fisheye.fisheye(R)
F.set_focus(focus)


fig, ax = pl.subplots(1,4,figsize=(13,5))
data = pl.imread('../sandbox/example.png')
ax[0].imshow(data)
ax[0].axis('off')
ax[0].set_title('original')

modes = ['Sarkar', 'default', 'root' ]
ds = [3, 3, 3]
xws = [0, 0.25, 0.25]

for i in range(1,4):
    F.set_mode(modes[i-1])
    F.set_magnification(ds[i-1])
    F.set_demagnification_width(xws[i-1])

    transformed = fisheye.apply_to_image(data, F)

    ax[i].imshow(transformed)
    ax[i].axis('off')
    ax[i].set_title(modes[i-1])


fig.savefig('img_example.png',dpi=300)
pl.show()

