from quickdraw import QuickDrawData

# df = pd.read_json('../archive/finger.ndjson', lines=True)
# for d in df['drawing']:
#     print(d)

# print(df['drawing'][0])
# draw = (zip(polyline[0], polyline[1]) for polyline in dt)
# print(draw)
# plt.imshow(draw)
# plt.show()


qd = QuickDrawData()
anvil = qd.get_drawing("anvil")
print(anvil)
anvil.image.show()