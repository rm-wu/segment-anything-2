# %%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torchmetrics import JaccardIndex


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
    "/home/mereur1/projects/ocl/clip_seg/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
)
model_cfg = "sam2_hiera_t.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    # points_per_side=64,
    points_per_side=32,
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
print(clip_dirs)
frame_step = 2
num_frames = 5
display_matches = True

# %%
for clip_dir in clip_dirs:
    clip_name = clip_dir.name
    print(f"Processing clip: {clip_name}")
    frames = list(clip_dir.glob("*.jpg"))
    frames = sorted(frames)
    print(f"Number of frames: {len(frames)}")
    clip_segments = {}
    initial_segmentations = {}
    predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
    inference_state = predictor.init_state(video_path=str(clip_dir))

    for ann_frame_idx in range(0, len(frames), 2):
        print()
        print(f"Processing frame: {ann_frame_idx}")
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
        masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        initial_segmentations[ann_frame_idx] = masks

        frame_names = [
            p
            for p in os.listdir(clip_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(str(p).split("/")[-1].split(".")[0]))
        print(frame_names)

        # Compare the masks of the current frame with the one computed in previous frame
        if ann_frame_idx - frame_step >= 0:
            matches = []
            for src_idx, src_mask in clip_segments[ann_frame_idx - frame_step][
                ann_frame_idx
            ].items():
                src = torch.from_numpy(src_mask).squeeze()
                j_index = JaccardIndex(task="multiclass", num_classes=2)
                for dest_idx, dest_mask in enumerate(masks):
                    trg = torch.from_numpy(dest_mask["segmentation"])
                    j_idx_score = j_index(src, trg)
                    if j_idx_score > 0.8:
                        # matches.append((src_idx, dest_idx, j_idx_score))
                        matches.append((src_idx, dest_idx))
                        print(
                            f"{src_idx} from {ann_frame_idx - frame_step} vs",
                            f" {dest_idx} from {ann_frame_idx}:  {j_idx_score}",
                        )
                        print(f"Opposite Jaccard index: {j_index(trg, src)}")
                        if display_matches:
                            # Display matched object using ann_frame_idx
                            img = np.ones(
                                (
                                    dest_mask["segmentation"].shape[0],
                                    dest_mask["segmentation"].shape[1],
                                    4,
                                )
                            )
                            img[:, :, 3] = 0
                            # plt.figure(figsize=(20, 20))
                            plt.imshow(frame)

                            ax = plt.gca()
                            ax.set_autoscale_on(False)

                            m = dest_mask["segmentation"]
                            color_mask = np.concatenate([np.random.random(3), [0.5]])
                            img[m] = color_mask
                            print(dest_mask["point_coords"])
                            point_coords = np.array(dest_mask["point_coords"])
                            print(point_coords.shape)
                            ax.scatter(
                                point_coords[:, 0],
                                point_coords[:, 1],
                                color="green",
                                marker="*",
                                s=200,
                                edgecolor="white",
                                linewidth=1.25,
                            )
                            ax.imshow(img)
                            plt.axis("off")
                            plt.show()
                            plt.close()

                            # Display matched object using ann_frame_idx - frame_step
                            img = np.ones((src.shape[0], src.shape[1], 4))
                            img[:, :, 3] = 0
                            # plt.figure(figsize=(20, 20))
                            plt.imshow(frame)
                            ax = plt.gca()
                            ax.set_autoscale_on(False)
                            m = src
                            color_mask = np.concatenate([np.random.random(3), [0.5]])
                            img[m] = color_mask
                            ax.imshow(img)
                            plt.axis("off")
                            plt.show()
                            plt.close()
                            print("=" * 50)
            print(
                f"there are {len(matches)} matches between frame {ann_frame_idx - frame_step} and frame {ann_frame_idx}"
            )

        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
        inference_state = predictor.init_state(video_path=str(clip_dir))

        prompts = {}
        # if ann_frame_idx - frame_step >= 0:
        
        if False:
            print(f"There are this matches: {matches}")
            src_idxs = [t[0] for t in matches]
            dest_idxs = [t[1] for t in matches]

            for prev_frame_idx in range(0, ann_frame_idx + 1, frame_step):
                print(prev_frame_idx)
                print(len(initial_segmentations[prev_frame_idx]))
                # Add the masks from the previous frames
                for obj_id, mask in enumerate(
                    tqdm(initial_segmentations[prev_frame_idx], ncols=100)
                ):
                    # print(obj_id)
                    ann_obj_id = obj_id
                    segmentation = torch.from_numpy(mask["segmentation"])
                    prompts[ann_obj_id] = segmentation
                    f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=prev_frame_idx,
                        obj_id=ann_obj_id,
                        mask=segmentation,
                    )

            current_frame_masks_idx = [range(0, len(masks))]
            # Add the matched masks using the same obj_id of the previous frame
            # to ensure consistency between the previous objects
            for src_idx, dest_idx in matches:
                ann_obj_id = src_idx
                segmentation = torch.from_numpy(masks[dest_idx]["segmentation"])
                prompts[ann_obj_id] = segmentation
                f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=segmentation,
                )

            new_obj_curr_frame = len(masks) - len(matches)
            last_obj_id = len(clip_segments[ann_frame_idx - frame_step][0])
            for dest_idx in range(new_obj_curr_frame):
                if dest_idx not in dest_idxs:
                    ann_obj_id = last_obj_id + dest_idx
                    segmentation = torch.from_numpy(masks[dest_idx]["segmentation"])
                    prompts[ann_obj_id] = segmentation
                    f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        mask=segmentation,
                    )
            start_frame = 0
            end_frame = (
                ann_frame_idx + frame_step
                if ann_frame_idx + frame_step <= num_frames
                else num_frames
            )
            video_segments = {}
            start_frame = ann_frame_idx
            end_frame = ann_frame_idx + 1

            if ann_frame_idx - frame_step >= 0:
                start_frame = 0
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(
                    inference_state,
                    reverse=True,
                    start_frame_idx=ann_frame_idx,
                    max_frame_num_to_track=ann_frame_idx,
                ):
                    # print(f"{ann_frame_idx - frame_step} >= 0")
                    print(f"Propagate backward: {out_frame_idx}")
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

            if ann_frame_idx + frame_step <= num_frames:
                end_frame = ann_frame_idx + frame_step + 1
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(
                    inference_state,
                    reverse=False,
                    start_frame_idx=ann_frame_idx,
                    max_frame_num_to_track=frame_step,
                ):
                    # print(f"{ann_frame_idx + frame_step} <= {num_frames}")
                    # print()
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            clip_segments[ann_frame_idx] = video_segments

            vis_frame_stride = 1
            plt.close("all")
            seg_path = clip_dir / "new_seg"
            seg_path.mkdir(exist_ok=True, parents=True)
            for out_frame_idx in range(start_frame, end_frame, vis_frame_stride):
                print("Plotting frame: ", out_frame_idx)
                # plt.figure(figsize=(6, 4))
                # plt.title(f"frame {out_frame_idx}")
                plt.imshow(
                    Image.open(os.path.join(clip_dir, frame_names[out_frame_idx]))
                )
                # print(video_segments[out_frame_idx].keys())
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    # print(out_obj_id)
                    # print(out_mask)
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id, random_color=True)
                    # if out_obj_id > 19: break
                plt.axis("off")
                plt.savefig(
                    seg_path / f"{ann_frame_idx}_seg_{out_frame_idx}.jpg",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.show()
                plt.close("all")

            # clip_segments[ann_frame_idx] = video_segments
            print(clip_segments.keys())
            print(video_segments.keys())
            print()
            print("=" * 100)
        else:
            for dest_idx, src_mask in tqdm(
                enumerate(masks),
                ncols=100,
            ):
                ann_obj_id = dest_idx
                segmentation = torch.from_numpy(src_mask["segmentation"])
                prompts[ann_obj_id] = segmentation
                f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=segmentation,
                )
            video_segments = {}
            start_frame = ann_frame_idx
            end_frame = ann_frame_idx + 1

            if ann_frame_idx - frame_step >= 0:
                start_frame = ann_frame_idx - frame_step
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(
                    inference_state,
                    reverse=True,
                    start_frame_idx=ann_frame_idx,
                    max_frame_num_to_track=frame_step,
                ):
                    # print(f"{ann_frame_idx - frame_step} >= 0")
                    print(f"Propagate backward: {out_frame_idx}")
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

            if ann_frame_idx + frame_step <= num_frames:
                end_frame = ann_frame_idx + frame_step + 1
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(
                    inference_state,
                    reverse=False,
                    start_frame_idx=ann_frame_idx,
                    max_frame_num_to_track=frame_step,
                ):
                    # print(f"{ann_frame_idx + frame_step} <= {num_frames}")
                    print()
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            clip_segments[ann_frame_idx] = video_segments

            vis_frame_stride = 1
            plt.close("all")
            seg_path = clip_dir / "new_seg"
            seg_path.mkdir(exist_ok=True, parents=True)
            for out_frame_idx in range(start_frame, end_frame, vis_frame_stride):
                print("Plotting frame: ", out_frame_idx)
                # plt.figure(figsize=(6, 4))
                # plt.title(f"frame {out_frame_idx}")
                plt.imshow(
                    Image.open(os.path.join(clip_dir, frame_names[out_frame_idx]))
                )
                # print(video_segments[out_frame_idx].keys())
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    # print(out_obj_id)
                    # print(out_mask)
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id, random_color=True)
                    # if out_obj_id > 19: break
                plt.axis("off")
                plt.savefig(
                    seg_path / f"{ann_frame_idx}_seg_{out_frame_idx}.jpg",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.show()
                plt.close("all")

            print(clip_segments.keys())
            print(video_segments.keys())
            print()
            print("=" * 100)
    break


#         predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
#         inference_state = predictor.init_state(video_path=str(clip_dir))
#         prompts = {}

#         for idx, mask in tqdm(
#             enumerate(sorted(masks, key=(lambda x: x["area"]), reverse=True)), ncols=100
#         ):
#             ann_obj_id = idx
#             segmentation = torch.from_numpy(mask["segmentation"])
#             prompts[ann_obj_id] = segmentation
#             f_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
#                 inference_state=inference_state,
#                 frame_idx=ann_frame_idx,
#                 obj_id=ann_obj_id,
#                 mask=segmentation,
#             )


#         video_segments = {}  # video_segments contains the per-frame segmentation results
#         for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
#             inference_state, reverse=True
#         ):
#             video_segments[out_frame_idx] = {
#                 out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                 for i, out_obj_id in enumerate(out_obj_ids)
#             }

#         for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
#             inference_state
#         ):
#             video_segments[out_frame_idx] = {
#                 out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                 for i, out_obj_id in enumerate(out_obj_ids)
#             }
#         clip_segments[ann_frame_idx] = video_segments

#         # Visualize the segmentation results for this ann_frame_idx
#         vis_frame_stride = 1
#         plt.close("all")

#         for out_frame_idx in trange(0, len(frame_names), vis_frame_stride, ncols=100):
#             # plt.figure(figsize=(6, 4))
#             # plt.title(f"frame {out_frame_idx}")
#             plt.imshow(Image.open(os.path.join(clip_dir, frame_names[out_frame_idx])))
#             # print(video_segments[out_frame_idx].keys())
#             for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#                 # print(out_obj_id)
#                 # print(out_mask)
#                 show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
#                 # if out_obj_id > 19: break
#             plt.axis("off")
#             plt.savefig(clip_dir / f"seg_{out_frame_idx}.jpg", bbox_inches='tight', pad_inches=0)
#             plt.close("all")

# # %%

# %%
