<h1>VideoToJpg</h1>

<h2>Description: </h2>

<p>Convert video to jpg images</p>

<h2>Installation: </h2>

<p>Install OpenCV, rembg, mediapipe</p>

<h2>Usage: </h2>

usage: convert.py [-h] [--video VIDEO] [--blur_level BLUR_LEVEL] [--num_faces NUM_FACES] [--model MODEL] [--skip_frames SKIP_FRAMES]
                  [--faces_thresh FACES_THRESH] [--draw_faces | --no-draw_faces] [--detect_faces | --no-detect_faces]
                  [--annotate_blur | --no-annotate_blur] [--remove_blurred | --no-remove_blurred] [--remove_background | --no-remove_background]


<h3>Arguments</h3>

``--video``
------------------
Path to the video

``--skip_frame``
------------------
Uses Laplacian to get the less blurred frame each 'skip_frames'

``--remove_blurred``
------------------
Uses Laplacian to remove blurred frames

``--blur_level``
------------------
Threshold to select frames to be removed if --remove_blurred is active

``--annotate_blur``
------------------
Annotate Laplacian value on the saved frame

``--remove_blurred``
------------------
Saved the bounding boxed region of the foreground object detected on the frame

``--remove_background``
------------------
Saved the bounding boxed region of the foreground object detected on the frame.
It uses the <a href="https://github.com/danielgatis/rembg">rembg</a> libary.

``--model``
------------------
Selects the segmentation model to use in the <a href="https://github.com/danielgatis/rembg">rembg</a> libary.
Tested models: u2net, u2net_human_seg, isnet-general-use, and sam

``--detect_faces``
------------------
Save only frames with faces

``--num_faces``
------------------
Number of possible faces in the frame

``--faces_thresh``
------------------
Face detection confidence

``--draw_faces``
------------------
Draw faces on the saved frame