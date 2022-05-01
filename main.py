from itertools import count
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skimage 
from skimage import color
import time
#import skimage

from utils import sample_func


#For Question #1
def GaussianBlurImage(image, sigma):
    
    start_time = time.time()

    im = Image.open(image)
    im = np.asarray(im)

    dims = im.shape

    filter_size = 2 * int(sigma * 4 + 0.5) + 1

    im = np.pad(im, pad_width=((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)), mode='constant', constant_values=128).astype(np.float32)

    gauss_filter = np.zeros((filter_size, filter_size, 3), dtype=np.float32)

    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            gauss_filter[i, j, :] = np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
            #1.0 / (2 * np.pi * sigma ** 2) * 

    #gauss_filter = gauss_filter/np.sum(gauss_filter)
    gauss_filter[:, :, 0] = gauss_filter[:, :, 0]/np.sum(gauss_filter[:, :, 0])
    gauss_filter[:, :, 1] = gauss_filter[:, :, 1]/np.sum(gauss_filter[:, :, 1])
    gauss_filter[:, :, 2] = gauss_filter[:, :, 2]/np.sum(gauss_filter[:, :, 2])

    output = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
    row = []

    for a in range(dims[0]):
        for b in range(dims[1]):
            frame = im[a:a+filter_size, b:b+filter_size, :]
    
            #conv = frame* gauss_filter.reshape((*gauss_filter.shape, 1))
            conv = frame* gauss_filter

            sum = [0, 0, 0]
            sum[0] = np.sum(conv[:, :, 0])
            sum[1] = np.sum(conv[:, :, 1])
            sum[2] = np.sum(conv[:, :, 2])

            output[a][b] = sum

    #Prep image to be saved
    output = output.astype(np.uint8)
    output = Image.fromarray(output)

    #Save image
    output.save("1.png")

    #Print Execution time
    print("---%s seconds ---" % (time.time() - start_time))
    
    #Display image
    plt.imshow(output)
    plt.title('GaussianBlurred image')
    plt.show()
    
    return output


#For Question #2
def SeparableGaussianBlurImage(image, sigma):

    start_time = time.time()
    im = Image.open(image)
    im = np.asarray(im)

    dims = im.shape
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
    im = np.pad(im, pad_width=((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)), mode='constant', constant_values=200).astype(np.float32)
    gauss_filter = np.zeros((filter_size, 3), dtype=np.float32)
    
    for i in range(filter_size):

        x = i - filter_size // 2
        gauss_filter[i] = (1.0 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2)/(2 * sigma ** 2))

    output = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)

    for a in range(filter_size // 2, dims[0] + filter_size // 2):
        for b in range(dims[1]):
            
            frame = im[a, b:b+filter_size, :]
            conv = frame* gauss_filter #.reshape((*gauss_filter.shape, 1))

            sum = [0, 0, 0]
            sum[0] = np.sum(conv[:, 0])
            sum[1] = np.sum(conv[:, 1])
            sum[2] = np.sum(conv[:, 2])

            output[a - filter_size // 2][b] = sum


    input = output
    output = output.astype(np.uint8)
    output = Image.fromarray(output)

    output2 = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
    input = np.pad(input, pad_width=((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)), mode='constant', constant_values=200).astype(np.float32)

    for a in range(filter_size // 2, dims[1] + filter_size // 2):
        for b in range(dims[0]):
            
            frame = input[b:b+filter_size, a, :]
            conv = frame* gauss_filter

            sum = [0, 0, 0]
            sum[0] = np.sum(conv[:, 0])
            sum[1] = np.sum(conv[:, 1])
            sum[2] = np.sum(conv[:, 2])

            output2[b][a - filter_size // 2] = sum

    output2 = output2.astype(np.uint8)
    output2 = Image.fromarray(output2)
    output2.save("2.png")
    
    print("---%s seconds ---" % (time.time() - start_time))

    plt.imshow(output2)
    plt.title('SeparableGaussianBlurred image')
    plt.show()

    return output2

#For Question #3
def SharpenImage(image, sigma, alpha):
    
    GausBlurredIm = GaussianBlurImage(image, sigma)
    GausBlurredIm = np.asarray(GausBlurredIm)

    image = Image.open(image)
    im = np.asarray(image)

    finalImage = np.clip(im - alpha*np.clip((GausBlurredIm.astype(np.double) - im.astype(np.double)),a_min=0, a_max=255) , a_min=0, a_max=255).astype(np.uint8)

    plt.imshow(finalImage)
    plt.title('Sharpened image')
    plt.show()

    finalImage = Image.fromarray(finalImage)
    finalImage.save("4.png")

    return finalImage

#For Question #4
def SobelImage(image):

    image = Image.open(image)
    
    im = color.rgb2gray(image)

    im = np.asarray(im)

    dims = im.shape

    sobel_filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    #sobel_filter_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobel_filter_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    filter_size = 3
    filter_buf = filter_size // 2

    im = np.pad(im, pad_width=((filter_buf, filter_buf), (filter_buf, filter_buf)), mode='constant', constant_values=128).astype(np.float32)

    Magoutput = np.zeros((dims[0], dims[1]), dtype=np.float32)
    Oroutput = np.zeros((dims[0], dims[1]), dtype=np.float32)

    count = 0

    for a in range(dims[0]):
        for b in range(dims[1]):
            
            #Apply sobel filter to frames to get new pixel val
            frame = im[a:a+filter_size, b:b+filter_size]
            conv_x = np.sum(frame* sobel_filter_x)
            conv_y = np.sum(frame* sobel_filter_y)

            #Calculate each pixel of magnitude image
            Magoutput[a][b] = 100*(np.sqrt(conv_x ** 2 + conv_y ** 2))
              
            #0-180 degrees  
            if np.arctan2(conv_y, conv_x) > 0:
                Oroutput[a][b] = np.arctan2(conv_y, conv_x)*127/np.pi
            else:
                Oroutput[a][b] = ((2 * np.pi)+np.arctan2(conv_y, conv_x)) * (127/np.pi)

    #Display image

    finalMagOutputToSave = Magoutput.astype(np.uint8)
    finalMagImage = Image.fromarray(finalMagOutputToSave)
    finalMagImage.save("5a.png")

    plt.imshow(finalMagOutputToSave, cmap="gray")
    plt.title('Sobel Filter - Magnitude image')
    plt.show()

    finalOrOutputToSave = Oroutput.astype(np.uint8)
    finalOrImage = Image.fromarray(finalOrOutputToSave)
    finalOrImage.save("5b.png")

    plt.imshow(finalOrOutputToSave)
    plt.title('Sobel Filter - Orientation image')
    plt.show()

    return Magoutput, Oroutput


#For Question #5
def BilinearInterpolation(im, c, d):

    #Floor values to get ride of extra decimal component
    leftx = np.floor(c)
    topy = np.floor(d)

    #Get the decimal component
    a = c - float(leftx)
    b = d - float(topy)

    x = leftx.astype(np.uint16)
    y = topy.astype(np.uint16)

    #Calculate the pixel value
    pixel = (1-a)*(1-b)*im[y, x] + a*(1-b)*im[y, x+1] + (1-a)*b*im[y+1, x] + a*b*im[y+1, x+1]

    return pixel


#For second part of Question #5
def UpSample(image, scale):

    im = Image.open(image)
    im = np.asarray(im)

    dims = im.shape

    scaleddims = []
    scaleddims.append(dims[0]*scale)
    scaleddims.append(dims[1]*scale)
    scaleddims.append(dims[2])

    scaledIm = np.ones((scaleddims[0], scaleddims[1], scaleddims[2]), dtype=np.uint8)
    scaledIm2 = np.ones((scaleddims[0], scaleddims[1], scaleddims[2]), dtype=np.uint8)

    xcoeff = float(dims[1] - 2) / float(dims[1]*scale - 1)
    ycoeff = float(dims[0] - 2) / float(dims[0]*scale - 1)

    for a in range(scaleddims[0]):
        for b in range(scaleddims[1]):

            #Scale using nearest neighbor
            scaledIm[a, b] = im[a // scale, b // scale]

            #Scale using bilinearInterpolation
            scaledIm2[a, b] = BilinearInterpolation(im, float(b) * xcoeff, float(a)* ycoeff).astype(np.uint8)


    plt.imshow(scaledIm)
    plt.title('Scaled (Nearest neighbor) Image')
    plt.show()

    finalImage = Image.fromarray(scaledIm)
    finalImage.save("6a.png")

    plt.imshow(scaledIm2)
    plt.title('Scaled (interpolation) Image')
    plt.show()

    finalImage2 = Image.fromarray(scaledIm2)
    finalImage2.save("6b.png")

    return scaledIm, scaledIm2
    
#For written question #2
def downSample(image, downScale):

    im = Image.open(image)
    im = np.asarray(im)

    dims = im.shape


    scaleddims = []
    scaleddims.append(dims[0]//downScale)
    scaleddims.append(dims[1]//downScale)
    scaleddims.append(dims[2])

    scaledIm = np.ones((scaleddims[0], scaleddims[1], scaleddims[2]), dtype=np.uint8)
    scaledIm2 = np.ones((scaleddims[0], scaleddims[1], scaleddims[2]), dtype=np.uint8)

    #xcoeff = float(dims[1] - 2) / float(dims[1]*scale - 1)
    #ycoeff = float(dims[0] - 2) / float(dims[0]*scale - 1)

    for a in range(1, scaleddims[0]):
        for b in range(1, scaleddims[1]):

            scaledIm[a, b] = im[a * downScale - 1, b * downScale - 1]
            #scaledIm2[a, b] = BilinearInterpolation(im, float(b) * xcoeff, float(a)* ycoeff).astype(np.uint8)

    plt.imshow(scaledIm)
    plt.title('DownSampled')
    plt.show()

    return scaledIm

#For Question #6
def FindPeaksImage(image, thres):

    imMag, imOr = SobelImage(image)
    
    dims = imOr.shape

    output = np.zeros((dims[0], dims[1]), dtype=np.float32)

    for a in range(dims[0]-1):
        for b in range(dims[1]-1):
            
            degrees = (imOr[a, b] * 2 * np.pi)/255
            dx1 = np.cos(degrees)
            dy1 = np.sin(degrees)

            dx2 = -dx1
            dy2 = -dy1

            e0_x = np.clip(b+dx1, a_min=0, a_max=(dims[1]-1))
            e0_y = np.clip(a+dy1, a_min=0, a_max=(dims[0]-1))

            e1_x = np.clip(b+dx2, a_min=0, a_max=(dims[1]-1))
            e1_y = np.clip(a+dy2, a_min=0, a_max=(dims[0]-1))

            if imMag[a, b] > BilinearInterpolation(imMag, e0_x, e0_y) and imMag[a, b] > BilinearInterpolation(imMag, e1_x, e1_y) and imMag[a, b] > thres:
                output[a, b] = 255
            else:
                output[a, b] = 0
            output[a, b]

    output = output.astype(np.uint8)
    finalImage = Image.fromarray(output)
    finalImage.save("7.png")

    plt.imshow(output, cmap='gray')
    plt.title('FindPeaks Image')
    plt.show()

    return output


#For Question #7
def BilateralImage(image, sigmaS, sigmaI):
    
    start_time = time.time()

    #Open the image
    im = Image.open(image)
    #im = color.rgb2gray(im)
    im = np.asarray(im)
    #im = im[300:400, 300:400, :]

    dims = im.shape

    #Determine the filter size based on sigma
    filter_size = 2 * int(sigmaS * 4 + 0.5) + 1

    #Pad the original image
    im = np.pad(im, pad_width=((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)), mode='constant', constant_values=128).astype(np.float32)

    #Declare size of gaussian
    gauss_filter = np.zeros((filter_size, filter_size, 3), dtype=np.float32)

    intensity_filter = np.zeros((filter_size, filter_size,3), dtype=np.float32)

    dI = np.zeros((3), dtype=np.float32)

    frame = np.zeros((3), dtype=np.float32)

    #Declare size of Intensity gaussian
    final_filter = np.zeros((filter_size, filter_size, dims[2]), dtype=np.float32)

    #Based on sigma, generate the space Gaussian Filter
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            gauss_filter[i, j, :] = np.exp(-(x ** 2 + y ** 2)/(2 * sigmaS ** 2))

    output = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)

    #Go through each row
    for a in range(dims[0]):

        #Go through each column
        for b in range(dims[1]):
            
            #Get a frame from the image FSxFS
            frame = im[a:a+filter_size, b:b+filter_size, :]
    
            for i in range(filter_size):
                for j in range(filter_size):

                    dI = abs(frame[filter_size//2, filter_size//2, :] - frame[i, j, :])
                    intensity_filter[i, j] = np.exp(-(dI ** 2)/(2 * sigmaI ** 2))
                    
            final_filter = np.multiply(gauss_filter, intensity_filter)
            final_filter[:, :, 0] = final_filter[:, :, 0]/np.sum(final_filter[:, :, 0])
            final_filter[:, :, 1] = final_filter[:, :, 1]/np.sum(final_filter[:, :, 1])
            final_filter[:, :, 2] = final_filter[:, :, 2]/np.sum(final_filter[:, :, 2])
            final_filter = np.multiply(final_filter,frame)

            sum = [0, 0, 0]
            sum[0] = np.sum(final_filter[:, :, 0])
            sum[1] = np.sum(final_filter[:, :, 1])
            sum[2] = np.sum(final_filter[:, :, 2])

            output[a][b] = sum
        print("a", a)

    #Prep image to be saved
    
    output = output.astype(np.uint8)
    output = Image.fromarray(output)

    #Save image
    output.save("Q7.png")

    #Print Execution time
    print("---%s seconds ---" % (time.time() - start_time))
    
    #Display image
    plt.imshow(output)
    plt.title('Bilateral GaussianBlurred image')
    plt.show()
    
    return output

#For Question 8
def HoughTransform(image, thres, thres2):

    image = FindPeaksImage(image, thres)

    image = Image.asarray(image)

    dims = image.shape

    output = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
    

    xCoord = []
    yCoord = []

    #Iterate through all pixels in image
    for a in range(dims[0]):
        for b in range(dims[1]):

            #If a pixel is white
            if image[a, b] > 0:
                #For all possible degrees
                for c in range(180):
                    #Find slopes/intercepts
                    theta = (c * 2 * np.pi) / 180
                    d = b*np.cos(theta) + a*np.sin(theta)
                    H[d, c] += 1

                    #Find values that are relatively large
                    if(H[d, c] > thres2):
                        Hmax = H[d, c]
                        xCoord.pop(b)
                        yCoord.pop(a)

    #Plot the theoretical lines on the output image
    output[a, b] = 
    #Where ever there is a maxiumum, we can assume there is a line

    #

    #Prep image to be saved
    
    output = output.astype(np.uint8)
    output = Image.fromarray(output)

    #Save image
    output.save("Q8.png")

    #Print Execution time
    #print("---%s seconds ---" % (time.time() - start_time))
    
    #Display image
    plt.imshow(output)
    plt.title('HoughTransform image')
    plt.show()

    return output

if __name__ == '__main__':
    
    #Q1
    #GaussianBlurImage('Seattle.jpg', 4)

    #Q2
    #SeparableGaussianBlurImage('Seattle.jpg', 4)

    #Q3
    #SharpenImage('Yosemite.png', 1, 5)

    #Q4
    #SobelImage('LadyBug.jpg')
    
    #Q5
    #UpSample('Moire_small.jpg', 4)

    #Written Q2
    #blurredMoire = GaussianBlurImage('Seattle.jpg', 6)
    #downSample(blurredMoire, 8)

    #Written Q3
    #SobelImage('TightRope.png')

    #Q6
    #FindPeaksImage('Circle.png', 40)

    #Q7
    #BilateralImage('Seattle.jpg', 2, 80)

    #Q8


    #Q9 Progressive Filter Fun
    
    #Show 1    
    #blurred = GaussianBlurImage('Seattle.jpg', 4)
    #toSave = Image.fromarray(blurred)
    #blurred.save("show1slide1.jpg")

    #Dont forget to uncomment in sobel filter
    #blurred2 = FindPeaksImage(blurred, 5)
    #toSave = Image.fromarray(blurred2)
    #toSave.save("show1slide2.jpg")

    #blurred3 = GaussianBlurImage2(blurred2, 4)
    #toSave = Image.fromarray(blurred3)
    #blurred3.save("show1slide3.jpg")

    #blurred4 = FindPeaksImage2(blurred3, 5)
    #toSave = Image.fromarray(blurred4)
    #toSave.save("show1slide4.jpg")

    #blurred5 = GaussianBlurImage2(blurred4, 4)
    #toSave = Image.fromarray(blurred5)
    #blurred5.save("show1slide5.jpg")

    #blurred6 = FindPeaksImage2(blurred5, 5)
    #toSave = Image.fromarray(blurred6)
    #toSave.save("show1slide6.jpg")


    #Show 2
    #newIm = GaussianBlurImage('Seattle.jpg', 4)
    #newIm.save("show2slide1.jpg")
    #newIm = SharpenImage(newIm, 4, 5)
    #newIm.save("show2slide2.jpg")


    #Show 3
    #newIm, interNewIm = UpSample('TightRope.png', 4)
    #toSave = Image.fromarray(newIm)
    #toSave.save("show3slide1.jpg")
    
    #newIm = downSample(newIm, 4)
    #toSave = Image.fromarray(newIm)
    #toSave.save("show3slide2.jpg")
    
    #newIm = downSample(newIm, 4)
    #toSave = Image.fromarray(newIm)
    #toSave.save("show3slide3.jpg")
    
    #newIm, interNewIm = UpSample2(newIm, 4)
    #toSave = Image.fromarray(interNewIm)
    #toSave.save("show3slide4.jpg")
    
    #newIm = SharpenImage(interNewIm, 4, 5)
    #toSave = Image.fromarray(newIm)
    #newIm.save("show3slide5.jpg")