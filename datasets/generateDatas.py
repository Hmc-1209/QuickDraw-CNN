from quickdraw import QuickDrawDataGroup
import cv2
import matplotlib.pyplot as plt

# Generate datas from quickdraw libraries
anvils = QuickDrawDataGroup("anvil", max_drawings=10)
# Integer i for sequential file name
i = 0
for anvil in anvils.drawings:
    print(anvil)
    # Saving pictures
    anvil.image.save('anvil'+str(i)+'.png')
    i += 1

# Read the first picture and display it
pic = cv2.imread('anvil0.png')
print(pic)
plt.imshow(pic)
plt.show()
