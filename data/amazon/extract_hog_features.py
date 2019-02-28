import os
import cv2

if __name__ == "__main__":
	#generates 1764 image features
	winSize = (64,64)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	winStride = (8,8)
	padding = (8,8)
	locations = ((10,20),)

	f_out = open('amazon_hog_features.csv', 'w')

	for f_name in os.listdir('./images'):
		if f_name.endswith(".jpg"):
			im = cv2.imread('./images/'+f_name)
			h = hog.compute(im,winStride,padding,locations).flatten().tolist()
			print(f_name)
			f_out.write(f_name.split(".")[0]+","+",".join([str(x) for x in h])+"\n")


	f_out.close()

