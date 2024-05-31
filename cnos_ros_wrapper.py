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

class CNOS_ROS:
    def __init__(self, cfg: DictConfig):
        OmegaConf.set_struct(cfg, False)
        logging.info("Initializing logger, callbacks and trainer")

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

        rospy.init_node("cnos_detetction")
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
        print("req detection...")

        start_time = time.time()
        rgb = goal.rgb
        inference_dataset = InferenceDL("/code/rgb/001136.png") #TODO rewrite to pass image, check image format, else should be ready
        
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

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        self.server.set_succeeded(response)


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def main_app(cfg: DictConfig):
    CNOS_ROS(cfg)
    rospy.spin()

if __name__ == "__main__":
    main_app()

