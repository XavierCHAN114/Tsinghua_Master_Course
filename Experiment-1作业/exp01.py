import cv2
import numpy as np


def Dark_Channel(Image):
    img=np.array(Image)
    V1 = img.min(2) 
    V1 = zmMinFilterGray(V1)
    return V1

def zmMinFilterGray(Image, r=1):   
    return cv2.erode(Image,np.ones((2*r-1,2*r-1)))

def guidedfilter(Image, p, r, eps):
    height, width = Image.shape
    m_I = cv2.boxFilter(Image, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(Image * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(Image * Image, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * Image + m_b

def get_V1_A(Image, w, maxV1, r, eps):   
    V1 =  Dark_Channel(Image)
    cv2.imwrite("/home/czy/Course/DIP/Experiment-1/rainy2_dark.jpg", V1*255)
    V1 = guidedfilter(V1, zmMinFilterGray(V1,2), r, eps)     #使用引导滤波优化 
    cv2.imwrite("/home/czy/Course/DIP/Experiment-1/rainy2_matting.jpg", V1*255)
    bins = 2000
    hist =np.histogram(V1, bins=2000)
    d = np.cumsum(hist[0])/float(V1.size)
    for lmax in range(bins-1, 0, -1):  
        if d[lmax] <= 0.999:
            break
    A = np.mean(Image,2)[V1 >= hist[1][lmax]].max()
    V1 = np.minimum(V1*w, maxV1)
    cv2.imwrite("/home/czy/Course/DIP/Experiment-1/rainy2_mask.jpg", V1*255)
    return V1, A


def deHaze(Image, w=0.95, maxV1 = 0.6, r=50, eps=0.0001):
    Y = np.zeros(Image.shape)
    V1, A = get_V1_A(Image, w, maxV1, r, eps) #得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (Image[:,:,k] - V1) / (1-V1/A) #颜色校正
    Y = np.clip(Y, 0, 1)
    
    return Y
    




if __name__ == '__main__':

    imgFile = "/home/czy/Course/DIP/Experiment-1/rainy2.png"  
    img1 = cv2.imread(imgFile, flags=1)/255 
    img = deHaze(img1) 
    img = img * 255
    cv2.imwrite("/home/czy/Course/DIP/Experiment-1/rainy2_dehaze.png", img)




