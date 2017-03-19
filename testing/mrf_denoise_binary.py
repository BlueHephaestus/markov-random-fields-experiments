# Image denoising using MRF model
from PIL import Image
import numpy as np
import cv2
from pylab import * 

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


def main():
    # Read in image
    im=Image.open('lena512.bmp')
    im=np.array(im)
    im=where (im>100,1,0) #convert to binary image
    (h,w)=im.shape

    # Add noise
    src=im.copy()
    noise=np.random.rand(h,w)
    ind=where(noise<0.2)
    src[ind]=1-src[ind]
    #src = (np.random.rand(1280,1920)*2).astype(int)
    """
    Generate random noise src image
    """
    #src = (np.random.rand(300,300)*2).astype(int)

    """
    Denoise the src image with Markov Random Fields. 
    """
    dst = MRF_denoise(src)
    
    white_divider = np.ones((src.shape[0], 1), dtype=np.float32)
    comparison_img = get_concatenated_row((src.astype(np.float32), white_divider, dst.astype(np.float32)))
    disp_img_fullscreen(comparison_img)

def MRF_denoise(src):
    """
    Given a src image, looks though and executes MRFs on this image,
        returning the dst image.
    """
    h, w = src.shape
    dst = np.zeros_like(src)

    while(SNR(src,dst)>10):
        print SNR(src, dst)
        for i in range(h):
            for j in range(w):
                neighbors=get_neighbors(i,j,h,w)#Get indices of neighbors
                
                a = cost(1,src[i,j],src,neighbors)#cost 
                b = cost(0,src[i,j],src,neighbors)

                if a>b:
                    dst[i,j]=1
                else:
                    dst[i,j]=0
        #src=dst
        break
    print SNR(src, dst)
    return dst

def SNR(A,B):
        """
        Given two matrices of same size, 
        This returns the mean of the absolute value difference between a and b.
            So it's a way of measuring how different they are on the large scale.
        You could also do the squared difference, if you wished.
        """
        #return np.sum(np.abs(A-B))
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
    Arguments:
    """
        
    """
    Our cost parameters
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
    return alpha * kronecker_delta(dst_val,src_val) + beta * np.sum(kronecker_delta(dst_val,neighbor_vals))

main()
