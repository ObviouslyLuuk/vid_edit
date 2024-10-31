import os
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from time import time


def get_video_dims(vid_path:str):
    """
    Get the dimensions of a video

    @param vid_path: path to the video
    @return: width, height
    """
    assert os.path.exists(vid_path), f"File {vid_path} does not exist"
    # cap = cv2.VideoCapture(vid_path)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.release()
    command = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {vid_path}'
    output = subprocess.run(command, shell=True, capture_output=True)
    dimensions = output.stdout.decode().split('x')
    width, height = int(dimensions[0]), int(dimensions[1])
    return width, height


def get_video_fps(vid_path:str):
    """
    Get the frames per second of a video

    @param vid_path: path to the video
    @return: frames per second
    """
    assert os.path.exists(vid_path), f"File {vid_path} does not exist"
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def scale_video_dims(width, height, scale:float|int):
    """
    Scale the dimensions of a video

    @param width: original width
    @param height: original height
    @param scale: if float: scale factor to downscale the video,
        if int: smallest side of the video after downscaling
    @return: out_width, out_height
    """
    if isinstance(scale, float):
        out_width = int(width*scale)
        out_height = int(height*scale)
    elif isinstance(scale, int):
        if width < height:
            out_width = scale
            out_height = int(scale*height/width)
        else:
            out_height = scale
            out_width = int(scale*width/height)
    else:
        raise ValueError("scale must be float or int")
    
    # Make both dimensions divisible by 2
    if out_width%2 != 0:
        out_width += 1
    if out_height%2 != 0:
        out_height += 1

    return out_width, out_height


def downscale_video(vid_path:str, outdir:str=None, outname:str=None, scale:float|int=144):
    """
    Downscale a video using ffmpeg.

    @param vid_path: path to the video
    @param outdir: directory to save the downscaled video,
        if None: save in the same directory as the input video
    @param outname: name of the downscaled video,
        if None: add '_{smallest_side}p' to the input video name
    @param scale: if float: scale factor to downscale the video,
        if int: smallest side of the video after downscaling
    @return: outpath if successful, None otherwise
    """
    assert os.path.exists(vid_path), f"File {vid_path} does not exist"
    if outdir is None:
        outdir = os.path.dirname(vid_path)
    # Get input dimensions
    width, height = get_video_dims(vid_path)
    # Get output dimensions
    out_width, out_height = scale_video_dims(width, height, scale)
    # Get output path
    if outname is None:
        outname = os.path.basename(vid_path).split('.')[0] + f'_{min(out_width, out_height)}p.mp4'
    outpath = os.path.join(outdir, outname)
    if os.path.exists(outpath):
        os.system(f'rm {outpath}')
    # Downscale video
    print(f"Downscaling {vid_path} to {outpath}")
    start_time = time()
    command = f'ffmpeg -i {vid_path} -vf scale={out_width}:{out_height} {outpath}'
    output = subprocess.run(command, shell=True, capture_output=True)
    if output.returncode != 0:
        print(f"Error: {output.stderr}")
        return None
    print(f"Done in {time()-start_time:.2f} seconds")
    return outpath


def extract_frames(vid_path:str, outdir:str=None, fps:int=1, ext:str='jpg'):
    """
    Extract frames from video using ffmpeg, and make dir
    outdir in the same parent dir to save them into.
    If there are already files in outdir they are deleted.

    @param vid_path: path to the video
    @param outdir: directory to save the frames,
        if None: save in a 'frames' directory in the same directory as the input video
    @param fps: frames per second to extract,
        if None: extract all frames
    @param ext: extension of the frames
    @return: outdir if successful, None otherwise
    """
    assert os.path.exists(vid_path), f"File {vid_path} does not exist"
    if outdir is None:
        outdir = os.path.join(os.path.dirname(vid_path), 'frames')
    if os.path.exists(outdir):
        os.system(f'rm -r {outdir}')
    os.makedirs(outdir)
    # Extract frames
    print(f"Extracting frames from {vid_path}")
    start_time = time()
    if fps is None:
        command = f'ffmpeg -i {vid_path} {outdir}/frame_%04d.{ext}'
    else:
        command = f'ffmpeg -i {vid_path} -vf fps={fps} {outdir}/frame_%04d.{ext}'
    output = subprocess.run(command, shell=True, capture_output=True)
    if output.returncode != 0:
        print(f"Error: {output.stderr}")
        return None
    print(f"Done in {time()-start_time:.2f} seconds")
    return outdir


def openrgb(image):
    """
    Open an image using cv2 in RGB format

    @param image: image to open
    @return: image in RGB format
    """
    return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

def savergb(image, path):
    """
    Save an image using cv2 in RGB format

    @param image: image to save
    @param path: path to save the image
    """
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def process_frame_batches(indir:str, outdir:str, fn:callable, batch_size:int=10):
    """
    Process frames in batches using a function fn and save them in outdir.
    If there are already files in outdir they are deleted.

    @param indir: directory containing frames
    @param outdir: directory to save the processed frames
    @param fn: function to process the frames,
        should accept a list of frames and return a list of processed frames
    @param batch_size: number of frames to process at once
        if None: attempt to process all frames at once
    @return: outdir if successful, None otherwise
    """
    assert os.path.exists(indir), f"Directory {indir} does not exist"
    assert len(os.listdir(indir)) > 0, f"Directory {indir} is empty"
    if os.path.exists(outdir):
        os.system(f'rm -r {outdir}')
    os.makedirs(outdir)
    # Process frames
    frame_files = sorted([f for f in os.listdir(indir) if f.endswith('.jpg') or f.endswith('.png')])
    if batch_size is None:
        batch_size = len(frame_files)
    print(f"Processing {len(frame_files)} frames in {len(frame_files)/batch_size} batches")
    for i in tqdm(range(0, len(frame_files), batch_size)):
        batch_files = frame_files[i:i+batch_size]
        batch = [openrgb(os.path.join(indir, f)) for f in batch_files]
        processed_batch = fn(batch)
        for j, f in enumerate(batch_files):
            savergb(processed_batch[j], os.path.join(outdir, f))
    return outdir


def construct_video(indir:str, outpath:str, fps:int=30, src_vid_path:str=None):
    """
    Construct a video from frames in indir using ffmpeg.
    If outpath exists, the user is prompted to overwrite it.
    y to overwrite, n to exit.

    @param indir: directory containing frames
    @param outpath: path to save the video
    @param fps: frames per second of the video,
        if None: get fps from src_vid_path
    @param src_vid_path: path to the source video
        used for getting source fps if fps is None
    @return: outpath if successful, None otherwise
    """
    assert os.path.exists(indir), f"Directory {indir} does not exist"
    assert len(os.listdir(indir)) > 0, f"Directory {indir} is empty"
    if os.path.exists(outpath):
        print(f"File {outpath} already exists")
        overwrite = input("Overwrite? (y/n): ")
        if overwrite == 'y':
            os.system(f'rm {outpath}')
        else:
            print("Exiting")
            return None
    assert fps or src_vid_path, "fps or src_vid_path must be provided"
    if fps == None:
        fps = get_video_fps(src_vid_path)
    # Construct video
    print(f"Constructing video from {indir}")
    start_time = time()
    ext = os.listdir(indir)[0].split('.')[-1]
    command = f'ffmpeg -r {fps} -i {indir}/frame_%04d.{ext} -c:v libx264 -vf fps={fps} {outpath}'
    output = subprocess.run(command, shell=True, capture_output=True)
    if output.returncode != 0:
        print(f"Error: {output.stderr}")
        return None
    print(f"Done in {time()-start_time:.2f} seconds")
    return outpath


def combine_audio(vid_path:str, audio_path:str, outpath:str=None):
    """
    Combine a video with an audio file using ffmpeg.

    @param vid_path: path to the video
    @param audio_path: path to the audio file,
        if .mp4: extract audio from the video
    @param outpath: path to save the video with audio,
        if None: save as {vid_path}_audio.mp4
    @return: outpath if successful, None otherwise
    """
    assert os.path.exists(vid_path), f"File {vid_path} does not exist"
    assert os.path.exists(audio_path), f"File {audio_path} does not exist"
    if outpath is None:
        outpath = vid_path.split('.')[0] + '_audio.mp4'
    if os.path.exists(outpath):
        os.system(f'rm {outpath}')
    # Combine audio
    print(f"Combining {vid_path} with audio from {audio_path}")
    start_time = time()
    if audio_path.split('.')[-1] == 'mp4':
        command = f'ffmpeg -i {vid_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {outpath}'
    else:
        command = f'ffmpeg -i {vid_path} -i {audio_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {outpath}'
    output = subprocess.run(command, shell=True, capture_output=True)
    if output.returncode != 0:
        print(f"Error: {output.stderr}")
        return None
    print(f"Done in {time()-start_time:.2f} seconds")
    return outpath


if __name__ == "__main__":
    # Test functions
    dir_name = 'face_test'
    vid_path = f'{dir_name}/face_test.mp4'
    scale = 144
    w, h = get_video_dims(vid_path)
    print("video size: ", w, h)
    # print("fps: ", get_video_fps(vid_path))
    print("new size: ", scale_video_dims(w, h, scale))
    vid_path = downscale_video(vid_path, scale=scale)

    fps = 15
    outdir = f'{dir_name}/frames'
    ext = 'jpg'

    extract_frames(vid_path, outdir, fps, ext)

    indir = outdir
    outdir = f'{dir_name}/processed_frames'
    batch_size = 200

    def process_fn(frames):
        frames_np = np.array(frames) # (N, H, W, C)
        frames_bw = np.mean(frames_np, axis=-1, keepdims=True)
        return frames_bw.astype(np.uint8)
    
    process_frame_batches(indir, outdir, process_fn, batch_size)

    outpath = f'{dir_name}/processed_video.mp4'
    src_vid_path = vid_path

    construct_video(outdir, outpath, fps, src_vid_path=None)

    audio_path = None
    
    combine_audio(outpath, audio_path=src_vid_path)

