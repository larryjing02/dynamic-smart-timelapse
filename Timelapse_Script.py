# Version 2 Timelapse: Modified speed mode, altered buffer
import imutils
import cv2
import time

######################################################################################
################################# Set variables here #################################
######################################################################################

# Output video filename
FILENAME  = 'outputSAMPLElong2.avi'

# Camera variables:
# If motion detection triggered too easily, turn auto-exposure to False and set exposure lock below
AUTO_EXPOSURE = True

# Note: set camera exposure slightly lower (so frame looks darker) if motion detection triggers too easily
EXPOSURE_LOCK = -7

CAMERA_SOURCE = 0

# Minimum size of detected contour to trigger motion detection
# Increase if motion detection too sensitive, decrease if motion not being detected
MIN_CONTOUR_SIZE = 5000

# How many frames to look behind for comparision during motion detection
FRAME_COMPARISON_DISTANCE = 15

# Specify whether or not to draw detection boxes
DRAW_FACE_BOXES = True
DRAW_MOTION_BOXES = False

# Specify whether or not to display image as code is running
DISPLAY_IMAGE = True

# Speed at which different event type frames are grabbed
FACE_EVENT_PERIOD = 1
MOTION_EVENT_PERIOD = 2

######################################################################################
######################################################################################
######################################################################################

# Three states:
# 1. Idle (no motion detected)
# 2. Motion (movement is detected)
# 3. Person (face is detected)
state = 0
comparison_frames = [None for i in range(FRAME_COMPARISON_DISTANCE)]
ref = 0

def detectFaces(curFrame, frame):
	# Detect faces
	faces = face_cascade.detectMultiScale(curFrame, 1.1, 6)
	detected = False
	
	# Loop over the faces
	for (x, y, w, h) in faces:
		# If the contour is too small, ignore it
		if w*h < MIN_CONTOUR_SIZE:
			continue
			
		if not detected:
			cv2.putText(frame, "Face Detected", (20,20),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			detected = True
			if not DRAW_FACE_BOXES:
				break
			
		# Compute bounding box & draw on frame, update text
		if DRAW_FACE_BOXES:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		
	return detected

def detectMotion(refFrame, curFrame, frame):
	# Compute absolute difference between current frame and first frame
	frameDelta = cv2.absdiff(refFrame, curFrame)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# Dilate thresholded image to fill in holes
	thresh = cv2.dilate(thresh, None, iterations=2)

	# Find contours on thresholded image
	contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	detected = False
	
	# loop over the contours
	for c in contours:
		# If the contour is too small, ignore it
		if cv2.contourArea(c) < MIN_CONTOUR_SIZE:
			continue
			
		if not detected:
			cv2.putText(frame, "Motion Detected", (20,20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			detected = True
			if not DRAW_MOTION_BOXES:
				break

		# Compute bounding box & draw on frame, update text
		if DRAW_MOTION_BOXES:
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
	return detected

######################################################################################
######################################################################################
######################################################################################

# Run Timelapse
# Initialize the face detecter
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video source - use webcam for testing
vs = cv2.VideoCapture(CAMERA_SOURCE)
# Set exposure
if AUTO_EXPOSURE:
	vs.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
else:
	vs.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_LOCK)

# Output will be written to output.avi
out = cv2.VideoWriter(
	'outputSAMPLElong2.avi',
	cv2.VideoWriter_fourcc(*'MJPG'),
	15.,
	(640,480))


cv2.startWindowThread()

# Allow time for source to begin recording (camera startup)
print("Starting video in 3... ",end="")
time.sleep(1)
print("2... ",end="")
time.sleep(1)
print("1... ")
time.sleep(1)

# Initialize dynamic frame grabbing variables
grab_period = MOTION_EVENT_PERIOD
elapsed_frames = 0
doubler = vs.get(cv2.CAP_PROP_FPS)

# Initialize circular array
for i in range(FRAME_COMPARISON_DISTANCE):
	ret, frame = vs.read()
	if ret:
		# TODO: Look into changing cv2 to imutils resize
		frame = cv2.resize(frame, (640, 480))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		comparison_frames[i] = cv2.GaussianBlur(frame, (41,41), 0)
		if i == FRAME_COMPARISON_DISTANCE - 1:
			if detectFaces(frame, frame):
				state = 3
			elif detectMotion(comparison_frames[0], comparison_frames[i], frame):
				state = 2
			else:
				state = 1
	else:
		print("Error with Video Capture Source")
		exit()

# Start normal loop, continues until terminated (press 'q')
current_frame = 0
while True:
	ret, frame = vs.read()
	if ret:
		# Ensure reference frame for motion detection is fine
		# TODO: Refactor to detectMotion method?
		if comparison_frames[ref] is None:
			print("Error with Comparison Frames")
			exit()
		
		# Advance frame counter
		current_frame += 1
		
		frame = cv2.resize(frame, (640, 480))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		if detectFaces(gray, frame):
			# Nonevent period just ended
			if state == 1:
				print("Face Detected, non-event period just ended!")
			state = 3
			if current_frame % FACE_EVENT_PERIOD == 0:
				out.write(frame.astype('uint8'))
			
		elif detectMotion(comparison_frames[ref], gray, frame):
			# Nonevent period just ended
			if state == 1:
				print("Face Detected, non-event period just ended!")
			state = 2
			if current_frame % MOTION_EVENT_PERIOD == 0:
				out.write(frame.astype('uint8'))
			
		else:
			# If state is still 2 or 3, event just ended (reset vars)
			if state > 1:
				print("Motion just ended, non-event period starting now!")
				grab_period = MOTION_EVENT_PERIOD
				elapsed_frames = 0
				doubler = vs.get(cv2.CAP_PROP_FPS)
			
			elapsed_frames += 1
			if elapsed_frames > doubler:
				doubler *= 2
				grab_period *= 2
				print(f"Current non-event playback speed: {grab_period}")

			if current_frame % grab_period == 0:
				cv2.putText(frame, f"{grab_period}x Speed", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				out.write(frame.astype('uint8'))	
			state = 1			
		
		# Update reference frame and circular array
		ref = (ref + 1) % FRAME_COMPARISON_DISTANCE
		if ref == 0:
			comparison_frames[-1] = gray
		else:
			comparison_frames[ref - 1] = gray

		# Display image
		if DISPLAY_IMAGE:
			cv2.imshow("Frame", frame)		
		
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	
vs.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)