from quickdraw import QuickDrawDataGroup
import cv2
import matplotlib.pyplot as plt

# df = pd.read_json('../archive/finger.ndjson', lines=True)
# for d in df['drawing']:
#     print(d)

# print(df['drawing'][0])
# draw = (zip(polyline[0], polyline[1]) for polyline in dt)
# print(draw)
# plt.imshow(draw)
# plt.show()


anvils = QuickDrawDataGroup("anvil", max_drawings=2)
i = 0
for anvil in anvils.drawings:
    print(anvil)
    anvil.image.save('anvil'+str(i)+'.png')
    i += 1

pic = cv2.imread('anvil0.png')
print(pic)
plt.imshow(pic)
plt.show()
