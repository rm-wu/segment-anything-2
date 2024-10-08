# %%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


from PIL import Image
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor

from utils import show_anns, show_points, show_mask


DATA_PATH = Path("/home/mereur1/projects/ocl/data/WTDataset/test/").resolve()
print(DATA_PATH)
assert DATA_PATH.exists(), "Data path does not exist"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

# Load SAM2 model
sam2_checkpoint = Path(
    "/home/mereur1/projects/ocl/clip_seg/segment-anything-2/checkpoints/sam2_hiera_large.pt"
)
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    # points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=400.0,
    use_m2m=True,
)

# Load clips directories
clip_dirs = sorted(list(DATA_PATH.glob("*/")))
# print(clip_dirs)

for clip_dir in clip_dirs:
    clip_name = clip_dir.name
    print(f"Processing clip: {clip_name}")
    frames = list(clip_dir.glob("*.jpg"))
    frames = sorted(frames)
    print(f"Number of frames: {len(frames)}")
    # Load the first frame

    ann_frame_idx = len(frames) // 2
    frame = Image.open(frames[ann_frame_idx])
    frame = np.array(frame.convert("RGB"))
    masks = mask_generator.generate(frame)
    print(len(masks))
    plt.figure(figsize=(20, 20))
    plt.imshow(frame)
    show_anns(masks)
    plt.axis("off")
    plt.show()
    torch.cuda.empty_cache()

    frame_names = [
        p
        for p in os.listdir(clip_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(str(p).split("/")[-1].split(".")[0]))
    # print(frame_names)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(clip_dir))
    prompts = {}

    for idx, mask in tqdm(
        enumerate(sorted(masks, key=(lambda x: x["area"]), reverse=True)), ncols=100
    ):
        ann_obj_id = idx
        segmentation = torch.from_numpy(mask["segmentation"])
        prompts[ann_obj_id] = segmentation
        f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=segmentation,
        )
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, reverse=True
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")

    for out_frame_idx in trange(0, len(frame_names), vis_frame_stride, ncols=100):
        # plt.figure(figsize=(6, 4))
        # plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(clip_dir, frame_names[out_frame_idx])))
        # print(video_segments[out_frame_idx].keys())
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # print(out_obj_id)
            # print(out_mask)
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            # if out_obj_id > 19: break
        plt.axis("off")
        plt.savefig(clip_dir / f"seg_{out_frame_idx}.jpg", bbox_inches='tight', pad_inches=0)
        plt.close("all")
