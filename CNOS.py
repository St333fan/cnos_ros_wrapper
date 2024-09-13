import logging
import multiprocessing
import os
import shutil
from functools import partial

import numpy as np
from PIL import Image
import torch
import distinctipy
from hydra.utils import instantiate
from tqdm import tqdm
from hydra import initialize, compose
import argparse
from src.model.utils import Detections
from src.utils.bbox_utils import CropResizePad
import glob
from pathlib import Path
from omegaconf import OmegaConf
import sys

from src.utils.visualization_utils import visualize_masks_multiple, visualize_masks_multiple_no_saving




# results = process_dataset(img_path,templates_dir, output_dir, args.stability_score_thresh, args.conf_threshold, args.light_itensity, args.radius)
class CNOSDetector:
  
  def __init__(self, 
               templates_dir,
               conf_threshold=0.5, 
               stability_score_thresh=0.97, 
               config_name="run_inference.yaml",
               subset=4):
    
    templates_dir = Path(templates_dir)
    
    with initialize(version_base=None, config_path="configs"):
      cfg = compose(config_name=config_name)
    
    cfg_segmentor = cfg.model.segmentor_model
    if "fast_sam" in cfg_segmentor._target_:
      logging.info("Using FastSAM, ignore stability_score_thresh!")
    else:
      cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    cfg.model.matching_config.confidence_thresh = conf_threshold
    
    logging.info("Initializing model")
    self.model = instantiate(cfg.model)
    
    self.model.ref_data = {}
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.descriptor_model.model = self.model.descriptor_model.model.to(self.device)
    self.model.descriptor_model.model.device = self.device
    
    # if there is predictor in the model, move it to device
    if hasattr(self.model.segmentor_model, "predictor"):
      self.model.segmentor_model.predictor.model = (
        self.model.segmentor_model.predictor.model.to(self.device)
      )
    else:
      self.model.segmentor_model.model.setup_model(device=self.device, verbose=True)
    
    logging.info(f"Moving models to {self.device} done!")
    
    self.templates_set = []
    templates_set = []
    for template_obj_dir in sorted(templates_dir.glob('*/')):
      template_name = os.path.split(template_obj_dir)[-1]
      logging.info(f"Initializing template {template_name}")
      template_paths = glob.glob(f"{template_obj_dir}/*.png")
      boxes, templates = [], []
      for idx, path in enumerate(template_paths):
        if idx % subset != 0:
          continue
        if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
          image = Image.open(path)
          boxes.append(image.getbbox())

          image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
          templates.append(image)

      if len(templates) == 0:
        logging.warn(f"No templates found in {template_obj_dir}")
        continue
      templates = torch.stack(templates).permute(0, 3, 1, 2)
      boxes = torch.tensor(np.array(boxes))

      processing_config = OmegaConf.create(
        {
          "image_size": 224,
        }
      )
      proposal_processor = CropResizePad(processing_config.image_size)
      templates = proposal_processor(images=templates, boxes=boxes).cuda()
      templates_set.append(templates)

    templates = torch.cat(templates_set)
    ref_feats = self.model.descriptor_model.compute_features(
      templates, token_name="x_norm_clstoken"
    )
    ref_feats = torch.stack(torch.chunk(ref_feats, len(templates_set)))
    self.model.ref_data["descriptors"] = ref_feats
    logging.info(f"Ref feats: {ref_feats.shape}")

    self.mask_vis_colors = distinctipy.get_colors(len(ref_feats))
    print("CNOS initialization finished!")
    
  
  def run_inference(self, rgb, output_dir=None):
      # run inference
    if output_dir is not None:
      rgb_name = "test"

    # run proposals
    detections = self.model.segmentor_model.generate_masks(np.array(rgb))

    # init detections with masks and boxes
    detections = Detections(detections)
    detections.remove_very_small_detections(
      config=self.model.post_processing_config.mask_post_processing
    )

    # compute descriptors
    query_decriptors = self.model.descriptor_model(np.array(rgb), detections)

    # matching descriptors
    (
      idx_selected_proposals,
      pred_idx_objects,
      pred_scores,
    ) = self.model.find_matched_proposals(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    detections.add_attribute("scores", pred_scores)
    detections.add_attribute("object_ids", pred_idx_objects)
    detections.apply_nms_per_object_id(
      nms_thresh=self.model.post_processing_config.nms_thresh
    )

    detections.to_numpy()
    masks = []
    if output_dir is not None:
      output_dir = Path(output_dir)
      
      if output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir()
        
      for i in range(detections.masks.shape[0]):
        mask = Image.fromarray(detections.masks[i].astype(np.uint8) * 255)
        mask.save(os.path.join(output_dir, f"{os.path.split(rgb_path)[-1][:-4]}_{detections.object_ids[i]}_{i}_mask.png"))
        masks.append(np.array(mask))
      visualize_masks_multiple(rgb, detections.masks.astype(bool), detections.object_ids, colors=self.mask_vis_colors, save_path=f"{output_dir}/{rgb_name}")
    
    else:
      for i in range(detections.masks.shape[0]):
        mask = Image.fromarray(detections.masks[i].astype(np.uint8) * 255)
        masks.append(np.array(mask))

    results = {"obj_ids": detections.object_ids,
                "masks": np.array(masks),
                "scores": detections.scores}
    
    return results

  
  def vis_results(self, rgb, results:dict):
    img = visualize_masks_multiple_no_saving(rgb, results['masks'].astype(bool), results['obj_ids'], colors=self.mask_vis_colors)
    return img


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_path", required=True, type=str, help="Path to the input image")
  parser.add_argument("--templates_dir", required=True, type=str, help="Path to the templates folder")
  parser.add_argument("--output_dir", default=None, type=str, help="Output directory for debug image")
  parser.add_argument("--conf_threshold", default=0.5, type=float, help="Confidence threshold")
  parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
  parser.add_argument("--light_itensity", default=1.0, type=float, help="Light itensity")
  parser.add_argument("--radius", default=0.4, type=float, help="Distance from camera to object")
  
  if len(sys.argv) == 1:
      args = parser.parse_args([
          '--image_path', '/mnt/Littleboy/cnos/datasets/tiny/scene/rgb/000005.png',
          '--templates_dir', '/mnt/Littleboy/cnos/datasets/tiny/templates_pyrender',
          '--output_dir', '/mnt/Littleboy/cnos/datasets/tiny/output_single_image',
          '--conf_threshold', '0.5',
          '--stability_score_thresh', '0.97',
          '--light_itensity', '1.0',
          '--radius', '0.4'
      ])
  else:
      args = parser.parse_args()
      
  
  detector = CNOSDetector(Path(args.templates_dir))
  
  results = detector.run_inference(args.image_path, args.output_dir)
  

