# Prepare thresholded points for DBSCAN clustering algorithm
from math import sqrt, ceil
from sklearn.cluster import DBSCAN
import scipy as sp
import hdbscan
import copy
import seaborn as sns
import io
import matplotlib.backend_bases as plt_base

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dp
import matplotlib 

matplotlib.use('Agg')
def imgShow(img):
    plt.imshow(img, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def imgsShow(images, size, labels=None, figsize=(14, 14)):
    fig = plt.figure(figsize=figsize)
    for i in range(1, size[0]*size[1] + 1):
        ax = fig.add_subplot(size[0], size[1], i)
        if labels:
            ax.set_title(labels[i - 1])
        plt.imshow(images[i-1], cmap="gray")
        plt.xticks([]), plt.yticks([])
    plt.show()
    
def getHist(img):
    first_channel = cv2.calcHist([img], [0], None, [256],[0,256])
    second_channel = cv2.calcHist([img], [1], None, [256],[0,256])
    third_channel = cv2.calcHist([img], [2], None, [256],[0,256])
    
    first_channel = np.reshape( first_channel, (256) ).astype(np.int64)
    second_channel = np.reshape( second_channel, (256) ).astype(np.int64)
    third_channel = np.reshape( third_channel, (256) ).astype(np.int64)
    
    return (first_channel, second_channel, third_channel)

def showHist(hist, figsize=(14, 5)):
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title("l channel")
    plt.plot(hist[0], color = (0, 0, 0))
    
    ax = fig.add_subplot(2, 3, 2)
    ax.set_title("a channel")
    plt.plot(hist[1], color = (0, 0, 0))
    
    ax = fig.add_subplot(2, 3, 3)
    ax.set_title("b channel")
    plt.plot(hist[2], color = (0, 0, 0))
    
    denseties = [ np.cumsum(h) for h in hist ]
    
    ax = fig.add_subplot(2, 3, 4)
    ax.set_title("l channel density")
    plt.plot(denseties[0], color = (0, 0, 0))
    
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title("a channel density")
    plt.plot(denseties[1], color = (0, 0, 0))
    
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title("b channel density")
    plt.plot(denseties[2], color = (0, 0, 0))
    
    plt.show()
    
def clusterize(input_img, color):
    img = color
    thresh_input = copy.deepcopy(input_img)
    #color_input = cv2.cvtColor(thresh_input, cv2.COLOR_GRAY2BGR)
    maxpoints=250000
    proxthresh=0.1

    binimg = thresh_input
    imgShow(binimg)
    X = np.transpose(np.where(binimg > 5))
    
    Xslice = X
    nsample = len(Xslice)
    if nsample > maxpoints:
        # Make sure number of points does not exceed DBSCAN maximum capacity
        Xslice = X[range(0,nsample,int(ceil(float(nsample)/maxpoints)))]

    # Translate DBSCAN proximity threshold to units of pixels and run DBSCAN
    pixproxthr = proxthresh * sqrt(binimg.shape[0]**2 + binimg.shape[1]**2)


    hdb = hdbscan.HDBSCAN(min_cluster_size=10,min_samples=50, cluster_selection_epsilon=pixproxthr).fit(Xslice)
    labels = hdb.labels_.astype(int)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Find the largest cluster (i.e., with most points) and obtain convex hull   
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(hdb.labels_, dtype=bool)
  
    #core_samples_mask[hdb.validity.internal_minimum_spanning_tree.internal_nodes] = True
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    fig = plt.figure(frameon=False)
    
    plot1 = fig.add_axes([0.,0.,1.,1.])
    plot1.axis('off')
    
    clusterer = hdb
    color_palette = sns.color_palette('deep', 8)
   
    cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for i, x in enumerate(clusterer.labels_)]

    cluster_member_colors = [sns.desaturate(x, p)
                             for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
    
    
    plot1.scatter(X[:,1],X[:,0], s=50, linewidth=0, c=cluster_member_colors, alpha=1)

    #input_image_rgb = copy.deepcopy(cv2.cvtColor(road, cv2.COLOR_RGB2BGR))
    input_image_rgb = color
    plot1.imshow(input_image_rgb)
   
    # define a function which returns an image as numpy array from figure
    def get_img_from_fig(fig, dpi=100):
        buf = io.BytesIO()
        
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    plot_img_np =  get_img_from_fig(fig)
    return plot_img_np
  
    
    
def makeVideo(filename, output):

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    cap = cv2.VideoCapture( filename + ".avi" )
    res_l = cv2.VideoWriter(output + "_l.avi" ,fourcc, 20.0, (240,140))
    res_b = cv2.VideoWriter(output + "_b.avi" ,fourcc, 20.0, (240,140))
    res_otsu = cv2.VideoWriter(output + "_o.avi" ,fourcc, 20.0, (240,140))
    res_mask = cv2.VideoWriter(output + "_m.avi" ,fourcc, 20.0, (240,140))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        l, _, b = cv2.split(frame[220:,200:440 ])
    
        mean, dev = cv2.meanStdDev( l )
        mean = mean[0]
        dev = dev[0]
        _, white = cv2.threshold( l, mean  + dev*(2+dev*2*np.sqrt(3)/(255)), 255, cv2.THRESH_BINARY)
        #white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)

        mean, dev = cv2.meanStdDev( b )
        mean = mean[0]
        dev = dev[0]
        _, yellow = cv2.threshold( b, mean  + dev*(2+dev*2*np.sqrt(3)/(255)), 255, cv2.THRESH_BINARY)
        yellow = cv2.cvtColor(yellow, cv2.COLOR_GRAY2BGR)
        
        _, otsu = cv2.threshold( b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

        
        #thresh_hel = cv2.resize(thresh_hel, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #road = cv2.resize(road, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        thresh_hel = white
        road = frame[220:, 200:440]
        mask = clusterize(thresh_hel, road)
        mask = cv2.resize(mask, (road.shape[1], road.shape[0]))
        
        white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
        res_l.write(white)
        res_b.write(yellow)
        res_otsu.write(otsu)
        res_mask.write(mask)
    
    res_l.release()
    res_b.release()
    res_otsu.release()
    cap.release()
    
video_path_res = "../images/video/res/"
video_path_clips = "../images/video/clips/"

makeVideo(video_path_clips + 'bright', video_path_res + 'bright')
makeVideo(video_path_clips + 'cloudy', video_path_res + 'cloudy')
makeVideo(video_path_clips + 'pale', video_path_res + 'pale')
makeVideo(video_path_clips + 'dark', video_path_res + 'dark')