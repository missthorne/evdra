
# this will test if the AI model trained properly
# todo
# FIX THIS SHITE

from PIL import Image
import numpy as np
import time
from collections import Counter

def num_detector(file_path):

    matched_arr = []
    load_examps = open('num_arr_ex.txt', 'r').read()
    load_examps = load_examps.split('\n')

    i = Image.open(file_path)
    iar = np.array(i)
    iarl = iar.tolist()

    in_question = str(iarl)

    for each_example in load_examps:
        try:
            split_ex = each_example.split('::')
            current_num = split_ex[0]
            current_arr = split_ex[1]

            each_pix_ex = current_arr.split('],')
            each_pix_in_q = in_question.split('],')

            x = 0

            while x < len(each_pix_ex):
                if each_pix_ex[x] == each_pix_in_q[x]:
                    matched_arr.append(int(current_num))

                x+=1

        except Exception as e:
            print(str(e))

    print(matched_arr)
    x = Counter(matched_arr)
    print(x)
    print(x[0])

num_detector('images/numbers/0.4.png')