# Assignment Week 2:
'''
You needn't finish reading all of them in just one week!
It's just good for you to know what's happening in this area and to figure out how people try to improve SIFT.

You needn't to remember all of them. 
But please DO REMEMBER procedures of SIFT and HoG. For those who're interested in SLAM, Orb is your inevitable destiny.

[Reading]:
1. [optional] Bilateral Filter: https://blog.csdn.net/piaoxuezhong/article/details/78302920
2. Feature Descriptors:
   [Compulsory]
   Hog: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
   SURF: https://www.vision.ee.ethz.ch/~surf/eccv06.pdf
   [optional]
   BRISK: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.1343&rep=rep1&type=pdf
   Orb: http://www.willowgarage.com/sites/default/files/orb_final.pdf [Compulsory for SLAM Guys]
3. Preview parts:
   K-Means: I have no doubts about what you are going to read and where you gonna find the reading materials. 
            There are tons of papers/blogs describing k-means. Just grab one and read.
			We'll this topic in 3 weeks.
			
[Coding]:			
1. 
#    Finish 2D convolution/filtering by your self. 
#    What you are supposed to do can be described as "median blur", which means by using a sliding window 
#    on an image, your task is not going to do a normal convolution, but to find the median value within 
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#
#    Python version:
#    def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#        Please finish your code under this blank
#
#
//   C++ version:
//   void medianBlur(vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way){
//       Please finish your code within this blank  
//   }

2. 【Reading + Pseudo Code】
#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A, 
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding 
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like: 
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#       
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while 
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And separated them by using a threshold 
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
#       def ransacMatching(A, B):
#           A & B: List of List
#
//      C++:
//      vector<vector<float>> ransacMatching(vector<vector<float>> A, vector<vector<float>> B) {
//      }    
#
#       Follow up 1. For step 3. How to do the "test“? Please clarify this in your code/pseudo code
#       Follow up 2. How do decide the "k" mentioned in step 5. Think about it mathematically!
#
# You are supposed to hand in the code in 1 week.
#


[Classical Project]
1. Classical image stitching!
   We've discussed in the class. Follow the instructions shown in the slides.
   Your inputs are two images. Your output is supposed to be a stitched image.
   You are encouraged to follow others' codes. But try not to just copy, but to study!
   
   You are supposed to hand in this project in 2-3 weeks.
			
'''
      
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import time

def medianBlur(img, kernel, padding_way):

    #    Finish 2D convolution/filtering by your self.
    #    What you are supposed to do can be described as "median blur", which means by using a sliding window
    #    on an image, your task is not going to do a normal convolution, but to find the median value within
    #    that crop.
    #
    #    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
    #    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO.
    #    img & kernel is List of List; padding_way a string

    #    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version
    #    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
    #    Follow up 1: Can it be completed in a shorter time complexity?
    img = np.array(img)
    W, H = img.shape
    s = 1  #kernel step
    kernel = np.array(kernel)
    img_o = img
    m, n = kernel.shape
    p_1 = max(((W - 1) * s + m - W), 0)
    pad_top = p_1 / 2
    pad_bottom = p_1 - pad_top
    p_2 = max(((H - 1) * s + n - H), 0)
    pad_left = p_2 / 2
    pad_right = p_2 - pad_left
    # Q = (n+2p-f)/s + 1
    if padding_way == "REPLICA":
        array_0_top = np.repeat(img[0, :].reshape((1, img.shape[1])), pad_top, axis=0)

        # print(array_0_top.shape)
        array_0_bottom = np.repeat(img[-1, :].reshape((1, img.shape[1])), pad_bottom, axis=0)
        img = np.insert(img, 0, array_0_top, axis=0)
        img = np.r_[img, array_0_bottom]
        array_0_right = np.repeat(img[:, -1].reshape((img.shape[0], 1)), pad_right, axis=1)

        img = np.c_[img, array_0_right]
        array_0_left = np.repeat(img[:, 0].reshape((img.shape[0], 1)), pad_left, axis=1)
        img = np.insert(img, 0, array_0_left.T, axis=1)
        print(img.shape)
        cv2.imshow("lenna_REPLICA_pad", img)
        cv2.waitKey()
    elif padding_way == "ZEROS":
        array_0_top = np.zeros((int(pad_top), img.shape[1]))
        array_0_bottom = np.zeros((int(pad_bottom), img.shape[1]))
        img = np.insert(img, 0, array_0_top, axis=0)
        img = np.r_[img, array_0_bottom]

        array_0_right = np.zeros((img.shape[0], int(pad_right)))
        img = np.c_[img, array_0_right]
        array_0_left = np.zeros((img.shape[0], int(pad_left)))
        img = np.insert(img, 0, array_0_left.T, axis=1)
        cv2.imshow("lenna_ZEROS_pad", img)
        cv2.waitKey()
    else:
        raise Exception("error padding way!")
    # # find the median blur
    # # 计算量为O[W*H*m*n*log(m*n)]
    # for i in range(W):
    #     for j in range(H):
    #         # img_out_array_tmp = img[i * m:(i+1) * m, j * n:(j+1) * m].flatten()
    #         left = i
    #         right = i + m
    #         bottom = j + n
    #         ceil = j
    #         # median_value = np.median(img[left:right, ceil:bottom])
    #         # img_o[i][j] = median_value
    #
    #         # 自己实现np.median
    #
    #         value_tmp = img[left:right, ceil:bottom].reshape((-1,1))
    #         array_1 = np.sort(value_tmp, axis=0)
    #         # print(type(array_1))
    #         # print(array_1.shape)
    #
    #         img_o[i][j] = array_1[int(m*n/2)][0]


    # # 如果需要减少计算量，就把二维卷积转化为两个一维的卷积，先对横轴做一维卷积，再对纵轴做一维卷积，
    # 这样相当于将原来的O[W*H*m*n*log(m*n)]的计算量减少为O[W*H*m*log(m) + W*H*n*long(n)]
    img_o_tmp = np.zeros(img.shape)
    for i in range(W):
        for j in range(H):

            bottom = j + n
            ceil = j
            # img_o_tmp[i, j] = np.median(img[i, ceil : bottom])
            value_tmp = img[i, ceil:bottom].reshape((-1,1))
            array_1 = np.sort(value_tmp, axis=0)
            # print(type(array_1))
            # print(array_1.shape)

            img_o_tmp[i][j] = array_1[int(n/2)][0]
    for i in range(W):
        for j in range(H):
            left = i
            right = i + m
            # img_o[i, j] = np.median(img_o_tmp[left : right , j])
            value_tmp_1 = img_o_tmp[left : right, j].reshape((-1, 1))
            array_2 = np.sort(value_tmp_1, axis=0)
            img_o[i, j] = array_2[int(m/2)][0]


    return img_o


def ransacMatching(A, B):
# A & B: List of List
# # PseudoCode
    iterations = 0
    best_model = null
    best_model_set = null
    best_error = 无穷大
    while(iterations < k || abs(best_error_pre - best_error_cur) < m)
        maybe_inliers = 从数据集中随机选择4对点
        maybe_model = 使用这4对点进行单应性矩阵的求解
        consensus_set = maybe_inliers
        for (每个数据集中不属于maybe_inliers的点)
            if (如果点适合于maybe_model，即错误小于t)
                将点添加到consensus_set
        #       这里test即为下文的单个点的this_error平方误差是否小于一定的阈值
        if (consensus_set中的点大于d)
            better_model = maybe_model
            this_error = better_model的误差度量
        if (this_error < best_error)
            best_model = better_model
            best_consensus_set = consensus_set
            best_error = this_error
        iterations++
    # this_error 计算方法为：
    # 假设计算出来的单应性矩阵为H，数据集中的点对为(x,y)->(x',y')，则
    # this_error = 数据集中所有点的 np.sum(np.square((np.dot(H,X) - Y))),即所有点的平方误差
    # k的计算为：k = log(1-p) / log(1-pow(w,n)), p 为模型适配的概率，w为局内点的概率，n为数据集中点的数目
    return best_model, best_consensus_set, best_error





img = cv2.imread("E://lenna.jpg",0)
cv2.imshow("lenna_origin",img)
cv2.waitKey()
kernel = [[0 for i in range(7)] for j in range(7)]
start_time = time.time()
img_median = medianBlur(img,kernel, "REPLICA")
end_time = time.time()
print(end_time - start_time)
cv2.imshow("lenna_median",img_median)
cv2.waitKey()
