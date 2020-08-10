import numpy as numpy
from skimage.measure import compare_ssim
import cv2
def PSNR(org, gen): 
    mse = np.mean((org - gen) ** 2) 
    if(mse == 0):   
        return 0
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def SSIM(org, gen):
    (score, diff) = compare_ssim(org1, gen1, full=True)
    return score


if __name__ == "__main__":
    imgOrg = cv2.imread('put',0)
    imgGen = cv2.imread('put',0)

    print(r"PSNR: {}".format(PSNR(imgOrg,imgGen)))
    print(r"SSIM: {}".format(SSIM(imgOrg,imgGen)))