import numpy as np
np.random.seed(420)
import cv2, h5py

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_concatenated_row(samples):
    """
    Concatenate each sample in samples horizontally, along axis 1.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=1)

def get_concatenated_col(samples):
    """
    Concatenate each sample in samples vertically, along axis 0.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=0)

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
            color = colors[int(prediction)]
            img[row_i*sub_h:row_i*sub_h+sub_h, col_i*sub_w:col_i*sub_w+sub_w] = color

def main():

    sub_h = 80
    sub_w = 145
    """
    sub_h = 6
    sub_w = 10
    """
    """
    sub_h = 33
    sub_w = 60
    """
    """
    sub_h = 1
    sub_w = 1
    """

    resize_factor = 1/8.
    alpha = 0.33
    epochs = 1
    class_n = 7
    colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]

    """
    Load a prediction from our predictions.h5, as our src.
    """
    with h5py.File("predictions.h5") as hf:
        src = np.array(hf.get("0"))

    h, w = src.shape

    """
    Denoise the src image with Markov Random Fields. 
    """
    dst = MRF_denoise(src, class_n, epochs)
    
    """
    For both our src and dst predictions,
        1. Create a zeroed img to store the colored overlay of predictions in,
        2. Generate an overlay with our zeroed img, predictions, and colors (and sub_h and sub_w),
        3. Then finally resize according to our resize_factor
    """
    """
    Open our greyscale and overlay src and dst onto their own versions.
    """
    with h5py.File("../tuberculosis_project/lira/lira1/data/greyscales.h5") as hf:
        img = np.array(hf.get("0"))
        img = img[0:h*sub_h, 0:w*sub_w]
        img = cv2.resize(img, (0,0), fx=resize_factor, fy=resize_factor)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        """
        src_img = np.zeros((h*sub_h, w*sub_w, 3), dtype=np.uint8)
        generate_overlay(src_img, src, colors, sub_h, sub_w)
        src_img = cv2.resize(src_img, (0,0), fx=resize_factor, fy=resize_factor)
        src_img = src_img.astype(np.uint8)
        src_img = cv2.addWeighted(src_img, alpha, img, 1-alpha, 0, img).astype(np.float32)
        """

        dst_img = np.zeros((h*sub_h, w*sub_w, 3), dtype=np.uint8)
        generate_overlay(dst_img, dst, colors, sub_h, sub_w)
        dst_img = cv2.resize(dst_img, (0,0), fx=resize_factor, fy=resize_factor)
        dst_img = cv2.addWeighted(dst_img, alpha, img, 1-alpha, 0, img).astype(np.float32)


    """
    Then concatenate the images horizontally,
        and display them.
    """
    #comparison_img = get_concatenated_col((src_img.astype(np.float32), dst_img.astype(np.float32)))
    cv2.imwrite("%i.jpg"%(epochs), dst_img)
    #disp_img_fullscreen(comparison_img)

def MRF_denoise(src, class_n, epochs):
    """
    Given a src image, looks though and executes MRFs on this image,
        returning the dst image.
    """
    h, w = src.shape

    MSE = lambda a, b: np.mean((a-b)**2)
    costs = np.zeros((class_n))

    for epoch in range(epochs):
        """
        Each loop, reset our destination image, which will contain the new denoised image.
        """
        dst = np.zeros_like(src)
        print MSE(src, dst)
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
        """
        Set src equal to dst, so that we can run this again on the result of the previous loop,
            and continue denoising the image for `epochs` times.
        """
        src = dst
    return dst


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
