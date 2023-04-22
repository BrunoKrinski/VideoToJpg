import os
os.environ["GLOG_minloglevel"] ="2"

import cv2
import argparse
import mediapipe as mp

from tqdm import tqdm
from pathlib import Path
from rembg import remove, new_session

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--blur_level', type=float)
    parser.add_argument('--num_faces', type=int, default=1)
    parser.add_argument('--model', type=str, default="u2net")
    parser.add_argument('--skip_frames', type=float, default=10)
    parser.add_argument('--faces_thresh', type=float, default=0.5)
    parser.add_argument('--draw_faces', action=argparse.BooleanOptionalAction)
    parser.add_argument('--detect_faces', action=argparse.BooleanOptionalAction)
    parser.add_argument('--annotate_blur', action=argparse.BooleanOptionalAction)
    parser.add_argument('--remove_blurred', action=argparse.BooleanOptionalAction)
    parser.add_argument('--remove_background', action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def remove_background(image, session):
    mask = remove(image, post_process_mask=True, session=session)
                            
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    value, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
        
    outputs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        size = w * h
        if size > 100000:
            outputs.append(image[y:y+h, x:x+w])
    return outputs

def detect_face(image, draw, thresh, num_faces):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=thresh)

    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        if draw:
            for detection in results.detections:
                x_min = int(detection.location_data.relative_bounding_box.xmin * image.shape[1])
                y_min = int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
                x_max = int((detection.location_data.relative_bounding_box.xmin + 
                             detection.location_data.relative_bounding_box.width) * image.shape[1])
                y_max = int((detection.location_data.relative_bounding_box.ymin + 
                             detection.location_data.relative_bounding_box.height) * image.shape[0])
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return True, image
    return False, image

def main():
    args = get_args()
    
    video_name = Path(args.video).stem
    os.makedirs(video_name, exist_ok=True)
    
    video = cv2.VideoCapture(args.video)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
    model_name = args.model        
    session = new_session(model_name)

    cont = 0
    frames = []
    laplaces = []
    for i in tqdm(range(total_frames)):

        success, frame = video.read()
        if not success: continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplace = cv2.Laplacian(gray, cv2.CV_64F).var()

        if len(frames) < args.skip_frames:
            frames.append(frame)
            laplaces.append(laplace)
        
        else:
            index = laplaces.index(max(laplaces))
            selected_frame = frames[index]
            selected_frame_laplace = laplaces[index]

            frames = []
            laplaces = []
            frames.append(frame)
            laplaces.append(laplace)

            if args.remove_blurred:
                if selected_frame < args.blur_level:
                    continue                    
                    
            if args.remove_background:
                outputs = remove_background(selected_frame, session)
            else:
                outputs = [selected_frame]

            save_frames = []
            if args.detect_faces:
                for output in outputs:
                    faces, output = detect_face(output,
                                                args.draw_faces,
                                                args.faces_thresh,
                                                args.num_faces)
                    if faces:
                        save_frames.append(output)
            else:
                save_frames = outputs

            for output in save_frames:
                if args.annotate_blur:
                    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                    laplace = cv2.Laplacian(gray, cv2.CV_64F).var()

                    cv2.putText(output, 
                                str(selected_frame_laplace), 
                                (50,50), 
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 3)

                pth = f'{video_name}/{video_name}_{str(cont)}.jpg'
                cv2.imwrite(pth, output)
                cont += 1
                            
if __name__ == '__main__':
    main()