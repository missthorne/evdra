

# This file shall be used as a tutorial area for image recognition systems in python
# Please wish me luck

#using pillow because 64bit

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean


def threshold(image_arr):
    balance_arr = [] # can't modify actual image array without numpy being angry, so this is used to average
    new_arr = image_arr # this is the array we will see with pixel vals
    for each_row in image_arr:
        for each_pix in each_row:
            avg_num = mean(each_pix[:3]) # each_pix :3 uwu
            balance_arr.append(avg_num) # read into what this does later

    balance = mean(balance_arr)
    for each_row in new_arr:
        for each_pix in each_row:
            if mean(each_pix[:3]) > balance: # :3, each furry comment speeds up exec time, trust.
                each_pix[0] = 255            # there is a reason why we run half the IT industry.
                each_pix[1] = 255            # if pixel is lighter than avg it's white
                each_pix[2] = 255            # otherwise it's black
                each_pix[3] = 255
            else:
                each_pix[0] = 0
                each_pix[1] = 0
                each_pix[2] = 0
                each_pix[3] = 255
    return new_arr


def create_examples():
    number_array_examples = open('num_arr_ex.txt', 'a')
    numbers_we_have = range(1,10)
    for each_num in numbers_we_have:
        #print each_num
        for further_num in numbers_we_have:
            # can just literally add *.1 and have it create float
            # but since we are gonna use it as string
            # this is aight
            print(str(each_num)+'.'+str(further_num))
            img_file_path = 'images/numbers/'+str(each_num)+'.'+str(further_num)+'.png'
            ei = Image.open(img_file_path)
            eiar = np.array(ei)
            eiarl = str(eiar.tolist())

            print(eiarl)
            line_to_write = str(each_num)+'::'+eiarl+'\n'
            number_array_examples.write(line_to_write)





# mapping them all to arrays
img1 = Image.open('images/numbers/0.1.png')
img1_arr = np.array(img1)
img2 = Image.open('images/numbers/y0.4.png') # tutorial is fucky, this does not work.
img2_arr = np.array(img2)
img3 = Image.open('images/numbers/y0.5.png')
img3_arr = np.array(img3)
img4 = Image.open('images/sentdex.png')
img4_arr = np.array(img4)

#applying_threshold
img1_arr = threshold(img1_arr)
img2_arr = threshold(img2_arr)
img3_arr = threshold(img3_arr)
img4_arr = threshold(img4_arr)


# preparing plot
fig = plt.figure()
ax1 = plt.subplot2grid((8,6),(0,0), rowspan=4, colspan=3)
ax2 = plt.subplot2grid((8,6),(4,0), rowspan=4, colspan=3)
ax3 = plt.subplot2grid((8,6),(0,3), rowspan=4, colspan=3)
ax4 = plt.subplot2grid((8,6),(4,3), rowspan=4, colspan=3)

# making sure it shows yeah
ax1.imshow(img1_arr)
ax2.imshow(img2_arr)
ax3.imshow(img3_arr)
ax4.imshow(img4_arr)

# actually showing the thing
plt.show()
create_examples()