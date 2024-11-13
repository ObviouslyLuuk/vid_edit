import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sam2.sam2_video_predictor import SAM2VideoPredictor
import argparse

# See if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

if device.type == "cuda":
    # Use bfloat16 for faster inference
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def cast_points_to_array(coords, labels):
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    coords = np.array(coords, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return coords, labels


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_frame_names(video_dir):
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names


def infer(points, labels):
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    return out_obj_ids, out_mask_logits

def update_plot(ax, img, points, labels, ann_frame_idx):
    ax.clear()
    ax.set_title(f"Frame {ann_frame_idx}")
    ax.imshow(img)
    if len(points) > 0:
        points, labels = cast_points_to_array(points, labels)
        show_points(points, labels, ax)
        out_obj_ids, out_mask_logits = infer(points, labels)
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])
    plt.draw()

def get_onclick(points, labels, ax, img, ann_frame_idx):
    def onclick(event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata

        # If there is already a point close to the clicked point, remove it
        # threshold is 2% of the width of the image
        threshold = 0.02 * img.width
        for i, (px, py) in enumerate(points):
            if (px - x)**2 + (py - y)**2 < threshold**2:
                points.pop(i)
                labels.pop(i)
                update_plot(ax, img, points, labels, ann_frame_idx)
                return
        
        if event.button == 1:
            onrightclick(x, y, points, labels)
        elif event.button == 3:
            onleftclick(x, y, points, labels)
        update_plot(ax, img, points, labels, ann_frame_idx)
    return onclick

def onrightclick(x, y, points, labels):
    points.append([x, y])
    labels.append(1)

def onleftclick(x, y, points, labels):
    points.append([x, y])
    labels.append(0)



if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./")
    parser.add_argument("--video_name", type=str, default="back_test")
    parser.add_argument("--frame_dirname", type=str, default="crop_frames")
    args = parser.parse_args()
    
    ROOT_DIR = args.root_dir
    VIDEO_NAME = args.video_name
    FRAME_DIR = os.path.join(ROOT_DIR, VIDEO_NAME, args.frame_dirname)
    print("Absolute path to frame directory: ", os.path.abspath(FRAME_DIR))
    frame_names = get_frame_names(FRAME_DIR)
    
    ann_frame_idx = 24  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    img = Image.open(os.path.join(FRAME_DIR, frame_names[ann_frame_idx]))

    # Make tmp dir with just the frame we are working with
    tmp_dir = os.path.join(ROOT_DIR, VIDEO_NAME, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    img.save(os.path.join(tmp_dir, frame_names[ann_frame_idx]))
    # Make tiny dummy frames in that directory up to the frame we are working with
    dummy_img = Image.new('RGB', img.size, color = 'red')
    for i in range(ann_frame_idx):
        dummy_img.save(os.path.join(tmp_dir, frame_names[i]))


    INFERENCE_STATE_PATH = os.path.join(ROOT_DIR, VIDEO_NAME, "sam_init_inference_state.pth")
    PROMPTS_PATH = os.path.join(ROOT_DIR, VIDEO_NAME, "sam_prompts.pth")

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    predictor = predictor.to(device)

    inference_state = predictor.init_state(
        video_path=tmp_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )

    points = []
    labels = []

    # show the results on the current (interacted) frame
    fig, ax = plt.subplots(figsize=(9, 9))
    # Listen for mouse click events
    fig.canvas.mpl_connect('button_press_event', get_onclick(points, labels, ax, img, ann_frame_idx))

    update_plot(ax, img, points, labels, ann_frame_idx)
    plt.show()

    # Save the state to a file
    points, labels = cast_points_to_array(points, labels)
    prompts = [
        {
            "frame_idx": ann_frame_idx,
            "obj_id": ann_obj_id,
            "points": points,
            "labels": labels,
        }
    ]
    torch.save(prompts, PROMPTS_PATH)
    # torch.save(inference_state, INFERENCE_STATE_PATH)

    # Remove the tmp dir and its contents
    os.system(f"rm -r {tmp_dir}")
