# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--out", required=False,
	help="path to store output image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=800)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = rect_to_bb(rect)
	#faceOrig = imutils.resize(image[y:y + h, x:x + w], width=800)
	faceAligned = fa.align(image, gray, rect)
	faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

	import uuid
	a = args["image"].split("/")
	if args["out"] != "":
		out_path = args["out"]
	else:
		out_path = "out/" + args["image"].split("/")[-1]

	cv2.imwrite(out_path, faceAligned)

	# display the output images
	#cv2.imshow("Original", faceOrig)
	#cv2.imshow("Aligned", faceAligned)
	#cv2.waitKey(0)
