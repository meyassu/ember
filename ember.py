import cv2
import numpy as np

def segment_flame(frame):
    """
    Segments the flame in a given frame using color-based thresholding.
    
    Args:
        frame (np.array): The frame in which flames need to be segmented.
        
    Returns:
        np.array: Binary mask where the flame pixels are set to 1.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for flame color in HSV
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([35, 255, 255])
    
    # Threshold the HSV image to get only flame colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Optional: Apply morphological operations to clean the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def process_video(video_path, output_path):
    """
    Reads the video, segments the flame in each frame, and writes the output video.
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to the output video.
    """
    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        return

    # Get video properties: width, height and frames per second
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Segment the flame in the current frame
        flame_mask = segment_flame(frame)

        # Overlay the mask on the original frame for visualization
        colored_mask = cv2.cvtColor(flame_mask, cv2.COLOR_GRAY2BGR)
        colored_mask[np.where((colored_mask == [255,255,255]).all(axis = 2))] = [0,0,255]
        
        output_frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        
        # Write the frame to the output video
        out.write(output_frame)

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    INPUT_VIDEO = "path_to_input_video.mp4"
    OUTPUT_VIDEO = "path_to_output_video.mp4"
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)
