from quickdraw import QuickDrawDataGroup
import cv2
import matplotlib.pyplot as plt

anvils = QuickDrawDataGroup("anvil", max_drawings=2)
i = 0
for anvil in anvils.drawings:
    print(anvil.image_data)
    anvil.get_image()
    # anvil.image.save('anvil'+str(i)+'.png')
    i += 1
#
# pic = cv2.imread('anvil0.png')
# print(pic)
# plt.imshow(pic)
# plt.show()

# t = [[[32]*32]*32]*100000
# print(t[10])
