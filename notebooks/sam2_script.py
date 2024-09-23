#%%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor


DATA_PATH = Path("./venice_short/").resolve()
print(DATA_PATH.exists())
#%%

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

def show_anns(anns, borders=True, marker_size=200):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
        print(ann['point_coords'])
        point_coords = np.array(ann['point_coords'])
        print(point_coords.shape)
        ax.scatter(point_coords[:, 0], point_coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    ax.imshow(img)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# %%
start_idx = 1000
first_frame_path = DATA_PATH / f"{start_idx:08d}.jpg"
first_frame = Image.open(first_frame_path)
first_frame = np.array(first_frame.convert("RGB"))

plt.figure(figsize=(20, 20))
plt.imshow(first_frame)
plt.axis('off')
plt.show()
# %%

sam2_checkpoint = Path("/home/mereur1/projects/ocl/clip_seg/segment-anything-2/checkpoints/sam2_hiera_small.pt")
model_cfg = "sam2_hiera_s.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    # points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=400.0,
    use_m2m=True,
)
masks = mask_generator.generate(first_frame)
print(len(masks))
# plt.figure(figsize=(20, 20))
# plt.imshow(first_frame)
# show_anns(masks2)
# plt.axis('off')
# plt.show() 
# %%


print(masks[0].keys())
points = []
for mask in masks:
    points.append(mask["point_coords"])
points = np.concatenate(points)
labels=np.ones(points.shape[0])
print(points.shape)
print(len(masks))
plt.figure(figsize=(20, 20))
plt.imshow(first_frame)
show_anns(masks)
# show_points(coords=points, labels=labels, ax=plt.gca())
plt.axis('off')
plt.savefig(f'./results/seg_sam2_ref.png')
plt.show()

#%%
anns = masks
borders = True
marker_size = 200
assert len(anns) != 0    
sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
# ax = plt.gca()
# ax.set_autoscale_on(False)


# for ann in sorted_anns:
#     fig, ax = plt.subplots(1)
#     ax.imshow(first_frame)
#     img = np.ones((sorted_anns[0]['segmentation'].shape[0],
#                    sorted_anns[0]['segmentation'].shape[1],
#                    4))
#     m = ann['segmentation']
#     color_mask = np.concatenate([np.random.random(3), [0.5]])
#     print(color_mask)
#     img[m] = color_mask
#     ax.imshow(img)
#     plt.axis('off')

#     plt.show()
    
#%%%
from torchmetrics import JaccardIndex 



for src_idx in range(len(masks)):
    # src_idx = 0
    src = torch.from_numpy(sorted_anns[src_idx]['segmentation'])


    j_index = JaccardIndex(task="multiclass", num_classes=2)
    for idx, ann in enumerate(sorted_anns):
        trg = torch.from_numpy(ann['segmentation'])
        j_idx_score = j_index(src, trg)
        if j_idx_score > 0.8 and idx != src_idx:
            print(f'{src_idx} vs {idx}:  {j_idx_score}')

        
#%%

# # Display single objects detected
# for ann in sorted_anns[0:12]:
#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0
#     plt.figure(figsize=(20, 20))
#     plt.imshow(first_frame)
    
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
    
#     m = ann['segmentation']
#     color_mask = np.concatenate([np.random.random(3), [0.5]])
#     img[m] = color_mask 
#     if borders:
#         import cv2
#         contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#         # Try to smooth contours
#         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#         cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
#     print(ann['point_coords'])
#     point_coords = np.array(ann['point_coords'])
#     print(point_coords.shape)
#     ax.scatter(point_coords[:, 0], point_coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.imshow(img)
#     plt.axis('off')
#     plt.show()
#     plt.close()

# ax.imshow(img)



#%%
del mask_generator
torch.cuda.empty_cache()


#######################################################################################################


#%%
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(DATA_PATH)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(frame_names)
inference_state = predictor.init_state(video_path=str(DATA_PATH))

#%%
prompts = {}
ann_frame_idx = 0
from tqdm import tqdm

for idx, mask in tqdm(enumerate(masks)):
    
    # # ADD THE ANCHOR POINTS
    # points_mask = np.array(mask['point_coords'])
    # labels = np.ones(points_mask.shape[0])
    # ann_obj_id = idx

    # prompts[ann_obj_id] = points_mask, labels
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=ann_obj_id,
    #     points=points_mask,
    #     labels=labels,
    #     # box=np.array(mask["bbox"])
    # )
    # print(points_mask)
    ann_obj_id = idx
    # print(type(mask['segmentation']))
    segmentation = torch.from_numpy(mask['segmentation'])
    prompts[ann_obj_id] = segmentation
    f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=segmentation 
    )


    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(Image.open(os.path.join(DATA_PATH, frame_names[ann_frame_idx])))
    # # show_points(points_mask, labels, plt.gca())
    # print(prompts)
    # for i, out_obj_id in enumerate(out_obj_ids):
    #     # show_points(*prompts[out_obj_id], plt.gca())
    #     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    # plt.show()
# %%
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(DATA_PATH, frame_names[out_frame_idx])))
    print(video_segments[out_frame_idx].keys())
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        # print(out_obj_id)
        # print(out_mask)
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        if out_obj_id > 19: break 
    plt.savefig(f'./results/seg_sam2_{out_frame_idx}.png')
    plt.show()
# %%
