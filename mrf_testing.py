import numpy as np
np.random.seed(420)
import cv2

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_concatenated_row(samples):
    """
    Concatenate each sample in samples horizontally, along axis 1.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=1)

def generate_overlay(img, predictions, colors, sub_h, sub_w):
    """
    Arguments:
        img: a np array of shape (h, w, 3),
            h is obtained by prediction_h * sub_h
            w is obtained by prediction_w * sub_w
            3 is color channel.
        predictions: an int np array of shape (prediction_h, prediction_w)
            with each entry being an index for the colors array, a prediction.
        colors: a list of colors for each prediction, a color key.
        sub_h, sub_w: heights of our subsections to assign each prediction to.

    Modifies the given image by placing rectangles of the color specified by each index in the predictions array,
        so that the resulting image is a bunch of colored rectangles of size sub_h x sub_w

    This is useful when we have predictions that are scalars for large areas, such as 80x145, and want to scale them up to overlay them.
    """

    for row_i, row in enumerate(predictions):
        for col_i, prediction in enumerate(row):
            color = colors[prediction]
            img[row_i*sub_h:row_i*sub_h+sub_h, col_i*sub_w:col_i*sub_w+sub_w] = color

def main():

    sub_h = 80
    sub_w = 145
    resize_factor = .1
    class_n = 7
    h = 100
    w = 25
    colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]

    """
    Generate a random noise src image given our h, w, and class_n
    """
    src = (np.random.rand(h,w)*(class_n)).astype(int)

    """
    Denoise the src image with Markov Random Fields. 
    """
    dst = MRF_denoise(src, class_n)
    
    """
    For both our src and dst predictions,
        1. Create a zeroed img to store the colored overlay of predictions in,
        2. Generate an overlay with our zeroed img, predictions, and colors (and sub_h and sub_w),
        3. Then finally resize according to our resize_factor
    """
    src_img = np.zeros((h*sub_h, w*sub_w, 3), dtype=np.uint8)
    generate_overlay(src_img, src, colors, sub_h, sub_w)
    src_img = cv2.resize(src_img, (0,0), fx=resize_factor, fy=resize_factor)

    dst_img = np.zeros((h*sub_h, w*sub_w, 3), dtype=np.uint8)
    generate_overlay(dst_img, dst, colors, sub_h, sub_w)
    dst_img = cv2.resize(dst_img, (0,0), fx=resize_factor, fy=resize_factor)

    """
    Then generate a divider for the images, 
        concatenate them horizontally,
        and display them.
    """
    white_divider = np.ones((src_img.shape[0], sub_w*resize_factor, 3), dtype=np.float32)
    comparison_img = get_concatenated_row((src_img.astype(np.float32), white_divider, dst_img.astype(np.float32)))
    disp_img_fullscreen(comparison_img)

def MRF_denoise(src, class_n):
    """
    Given a src image, looks though and executes MRFs on this image,
        returning the dst image.
    """
    h, w = src.shape
    dst = np.zeros_like(src)

    costs = np.zeros((class_n))
    #while(SNR(src,dst)>10):
    for i in range(20):
        print SNR(src, dst)
        for i in range(h):
            for j in range(w):

                """
                Get indices of neighbors
                """
                neighbors=get_neighbors(i,j,h,w)
                
                """
                Get cost of each class for this pixel in our src img
                """
                for class_i in range(class_n):
                    costs[class_i] = cost(class_i,src[i,j],src,neighbors)

                """
                Assign dst pixel to class with highest cost.
                """
                dst[i,j] = np.argmax(costs)
        src = dst
        dst = np.zeros_like(src)
        #src=dst
        #break
    print SNR(src, dst)
    return dst

def SNR(A,B):
        """
        Given two matrices of same size, 
        This returns the mean of the absolute value difference between a and b.
            So it's a way of measuring how different they are on the large scale.
        You could also do the squared difference, if you wished.
        """
        #return np.mean(np.abs(A-B))
        return np.sum(np.square(A-B))

def kronecker_delta(a,b):
        """
        The Kronecker Delta function is really useful, but there isn't an actual method in many libraries.
        Fortunately, it's pretty much just ~(a-b), since we want the following behavior:
            kronecker_delta(a,b) = 1 if a == b
            kronecker_delta(a,b) = 0 if a != b
        So we do this, then return the integer representation of our logical op.
        Since I use numpy it holds for scalars and also vectors/matrices.
        """
        return np.logical_not(a-b).astype(int)

def get_neighbors(i,j,h,w):
    """
    Get all adjacent neighbors, vertically, diagonally, and horizontally.
    We handle our edge cases by getting all 8 of these neighbors, then looping backwards through the list
        and removing those that aren't inside the bounds of our image.
    """
    neighbors=[(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
    for neighbor_i in range(len(neighbors)-1, -1, -1):#Iterate from len-1 to 0
        sample_i, sample_j = neighbors[neighbor_i]
        if sample_i < 0 or sample_i > h-1 or sample_j < 0 or sample_j > w-1:
            del neighbors[neighbor_i]
    return neighbors

def cost(dst_val,src_val,src,neighbors):
    """
    Our cost parameters
        Apparently when A = 10*B, dst == src
    """
    alpha=1
    beta=10

    """
    The values of the neighbor indices of our src pixel. We get these as a vector to speed up the cost computation.
    """
    neighbor_vals = np.array([src[neighbor] for neighbor in neighbors])

    """
    Compute our cost function as follows, using our neighbor value vector to compute the neighbor kronecker deltas simultaneously.
    """
    return (alpha * kronecker_delta(dst_val,src_val) + beta * np.sum(kronecker_delta(dst_val,neighbor_vals)))

main()
