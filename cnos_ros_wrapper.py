import os.path as osp
import os
import argparse
import time

import numpy as np

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import json
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
#import torch transforms
import torchvision.transforms as T
import numpy as np
from src.dataloader.bop import InferenceDL
import time
import ros_numpy
from sensor_msgs.msg import Image, RegionOfInterest

item_dict = {
    1: '002_master_chef_can',
    2: '003_cracker_box',
    3: '004_sugar_box',
    4: '005_tomato_soup_can',
    5: '006_mustard_bottle',
    6: '007_tuna_fish_can',
    7: '008_pudding_box',
    8: '009_gelatin_box',
    9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick'
}



class CNOS_ROS:
    def __init__(self, cfg: DictConfig):
        OmegaConf.set_struct(cfg, False)
        logging.info("Initializing logger, callbacks and trainer")
        self.idx = 0
        self.trainer = instantiate(cfg.machine.trainer)

        default_ref_dataloader_config = cfg.data.reference_dataloader
        default_query_dataloader_config = cfg.data.query_dataloader

        query_dataloader_config = default_query_dataloader_config.copy()
        ref_dataloader_config = default_ref_dataloader_config.copy()

        if cfg.dataset_name in ["hb", "tless"]:
            query_dataloader_config.split = "test_primesense"
        else:
            query_dataloader_config.split = "test"
        query_dataloader_config.root_dir += f"{cfg.dataset_name}"

        logging.info("Initializing model")
        model = instantiate(cfg.model)
        
        model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names
        model.dataset_name = cfg.dataset_name

        if cfg.model.onboarding_config.rendering_type == "pyrender":
            ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
            ref_dataset = instantiate(ref_dataloader_config)
        elif cfg.model.onboarding_config.rendering_type == "pbr":
            logging.info("Using BlenderProc for reference images")
            ref_dataloader_config._target_ = "src.dataloader.bop_pbr.BOPTemplatePBR"
            ref_dataloader_config.root_dir = f"{query_dataloader_config.root_dir}"
            ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
            ref_dataset = instantiate(ref_dataloader_config)
            ref_dataset.load_processed_metaData(reset_metaData=True)
        else:
            raise NotImplementedError
        model.ref_dataset = ref_dataset

        segmentation_name = cfg.model.segmentor_model._target_.split(".")[-1]
        agg_function = cfg.model.matching_config.aggregation_function
        rendering_type = cfg.model.onboarding_config.rendering_type
        level_template = cfg.model.onboarding_config.level_templates
        model.name_prediction_file = f"{segmentation_name}_template_{rendering_type}{level_template}_agg{agg_function}_{cfg.dataset_name}"
        logging.info(f"Loading dataloader for {cfg.dataset_name} done!")
        logging.info(f"---" * 20)
        self.model = model
        self.trainer =  self.trainer
        self.cfg = cfg

        rospy.init_node("cnos_detection")
        self.server = actionlib.SimpleActionServer('/object_detector/cnos',
                                                    GenericImgProcAnnotatorAction,
                                                    execute_cb=self.detect_objects,
                                                    auto_start=False)
        self.server.start()

        print("Object detection with CNOS is ready.")


    """
    When using the robokudo_msgs, as the callback function for the action server
    """
    def detect_objects(self, goal):
        print("Detecting Objects\n")
        
        start_time = time.time()
        rgb = goal.rgb

        rgb = ros_numpy.numpify(rgb) #TODO maybe set intrinsics somewhere, seems to work without???


        inference_dataset = InferenceDL(rgb, idx=self.idx)
        
        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=1,  # only support a single image for now
            num_workers=self.cfg.machine.num_workers,
            shuffle=False,
        )
        response = self.trainer.predict(
            self.model,
            dataloaders=inference_dataloader,
        )

        response = response[0]
        category_id = response['category_id']
        scores = response['score']
        bbox = response['bbox']
        mask = response['segmentation']

        # sort the four np.arrays based on the scores
        order = np.argsort(scores)[::-1]
        category_id = category_id[order]
        scores = scores[order]
        bbox = bbox[order]
        mask = mask[order]
        score = scores[0]
        bboxes = []
        label_image = np.full_like(mask[0],-1, dtype=np.int16)
        for i, score in enumerate(scores):
            if score < 0.5:
                break
            label_image[mask[i]>0] = i
            bb = RegionOfInterest()
            bb.x_offset = bbox[i][0]
            bb.y_offset = bbox[i][1]
            bb.width = bbox[i][2]
            bb.height = bbox[i][3]
            bboxes.append(bb)

        result = GenericImgProcAnnotatorResult()
        result.success = True
        result.bounding_boxes = bboxes
        result.class_confidences = scores[0:i]
        result.image = ros_numpy.msgify(Image, label_image, encoding='16SC1')
        result.class_names = [item_dict[i] for i in category_id[0:i]]

        print("\nDetected Objects:\n")
        print(result.class_names)
        print(result.class_confidences)
        

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        self.idx += 1
        self.server.set_succeeded(result)


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def main_app(cfg: DictConfig):
    CNOS_ROS(cfg)
    rospy.spin()

if __name__ == "__main__":
    main_app()

