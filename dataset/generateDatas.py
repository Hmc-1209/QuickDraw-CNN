"""
    To generate datas from quickdraw database, simply call generateData
    function and pass the keywords and values in, then the code will
    automatically generate pictures in this folder.

    .quickdrawcache is a cache when calling quickdraw library.
    You don't need to do anything about it.
"""

from quickdraw import QuickDrawDataGroup


def generateData(key, value):
    print('Generating data pictures ... ')
    # Generate datas from quickdraw libraries
    keys = QuickDrawDataGroup(key, max_drawings=value)
    # Integer i for sequential file name
    i = 0
    for element in keys.drawings:
        # Saving pictures
        element.image.save(key + str(i) + '.png')
        i += 1
    print('Pictures have been generated !')


"""
Generate data using the below form 
generateData('anvil', 100) --> # Representing generate 100 anvil pictures
"""

generateData('anvil', 10)

