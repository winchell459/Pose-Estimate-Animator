import cv2
import mediapipe as mp
import numpy as np

import os
import json

def pose_to_json(poses, folder, name):
    posesArray = []

    index = 0
    for pose in poses:
        poseDict = {}
        if (pose.pose_landmarks != None):
            landmarksDict = []

            for landmark in pose.pose_landmarks.landmark:
                poseLandmark = {}
                poseLandmark["x"] = landmark.x
                poseLandmark["y"] = landmark.y
                poseLandmark["z"] = landmark.z
                poseLandmark["visibility"] = landmark.visibility
                landmarksDict.append(poseLandmark)

            poseDict["landmarks"] = landmarksDict
            posesArray.append(poseDict)
        else:
            print("Missing pose_landmarks")

    poseDict = {}
    poseDict["poses"] = posesArray

    with open(folder + "//" + name, "w") as outfile:
        json.dump(poseDict, outfile)


def imageFileProcessing(path):
    IMAGE_FILES = []
    dir_list = os.listdir(path)
    for file in dir_list:
        if ('.meta' not in file): IMAGE_FILES.append(path + "//" + file)
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            poses.append(results)
            if not results.pose_landmarks:
                continue

            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            )


def videoFileProcessing(path):
    cap = cv2.VideoCapture(path)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            poses.append(results)
            if not results.pose_landmarks:
                continue

            image_height, image_width, _ = image.shape
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            )

    cap.release()


# ---------------------------------------- main ------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


poses = []
videoProcessing = True;
#location of images (directory) or the path to video
path = "Samples//vecteezy_woman-running-ocean-beach-young-asian-female-exercising_10012531_100.mov"
path = "Samples//findstory_video_116971_1080p_watermark.mp4"

if(videoProcessing):
    videoFileProcessing(path)
else:
    # For static images:
    imageFileProcessing("Samples//ezgif-4-51707fa179-jpg")

#directory to save json file
folder = 'Samples'
filename = 'running1.json'
pose_to_json(poses, folder, filename)