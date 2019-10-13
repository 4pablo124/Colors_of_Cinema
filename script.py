import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageStat


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=2):
            self.CLUSTERS = clusters
            self.IMAGE = image


    def getDominantColors(self):

        img = self.IMAGE          
        
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
            
        #save image after operations
        self.IMAGE = img

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)

        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        #returning after converting to integer from float
        return self.COLORS.astype(int)

    def plotHistogram(self,height=300, width=500):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((round(height*0.15), width, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * width
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), height), (r,g,b), -1)
            start = end	
        
        return chart

def is_monochromatic_image(img):
    extrema = img.convert("L").getextrema()
    return  (extrema[1] - extrema[0]) < 10

def extract_frames(path, interval_in_sec=1, clusters = 5):
    movie = cv2.VideoCapture(path)
    fps = round(movie.get(cv2.CAP_PROP_FPS))
    count = 0
    success= 1
    print("Extracting a frame every %d"%interval_in_sec+" seconds\n")
    while success:
        success,image = movie.read()

        if count%(fps*interval_in_sec) == 0:
            print("Frame: %d"%count)
            if is_monochromatic_image(Image.fromarray(image, 'RGB')):
                print("\tMonocromatic frame -> Skipping")
            else:
                print("\tCalculating...")
                dc = DominantColors(image,clusters)

                dc.getDominantColors()

                height, width = image.shape[:2]
                dc_image = dc.plotHistogram(height, width)

                final_image = np.concatenate((image,dc_image),axis=0)

                cv2.imwrite("frames/frame%d.jpg"%count, final_image)
                print("\tSaved on 'frames/frame%d.jpg'"%count)
            print("\nSearching next frame...\n")
        count+=1

extract_frames("Original.mp4",120)
