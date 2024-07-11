import numpy as np
import time
import torch
from PIL import Image as IM
import logging
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import argparse
import glob
from src.utils.bbox_utils import CropResizePad
from torchvision.utils import save_image
from src.model.utils import Detections
from src.model.loss import Similarity
import rospy
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
from omegaconf import OmegaConf
from hydra.utils import instantiate
#import torch transforms
import ros_numpy
from sensor_msgs.msg import Image, RegionOfInterest
from src.poses.pyrender import main as render
import os

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
    def __init__(self, stability_score_thresh, num_max_dets, conf_threshold, gpu_devices, cad_path, obj_pose, output_dir, light_itensity, radius):
        self.object_name = os.path.basename(cad_path).split(".")[0]
        self.num_max_dets = num_max_dets
        self.conf_threshold = conf_threshold
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name='run_inference.yaml')
        cfg_segmentor = cfg.model.segmentor_model
        if "fast_sam" in cfg_segmentor._target_:
            logging.info("Using FastSAM, ignore stability_score_thresh!")
        else:
            cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        self.metric = Similarity()
        logging.info("Initializing model")
        self.model = instantiate(cfg.model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(device)
        self.model.descriptor_model.model.device = device
        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=device, verbose=True)
        logging.info(f"Moving models to {device} done!")
        
        logging.info("Render Templates")
        render(gpu_devices, cad_path, obj_pose, output_dir, light_itensity, radius)

        logging.info("Initializing template")
        template_paths = glob.glob(f"{output_dir}/*.png")
        boxes, templates = [], []
        for path in template_paths:
            image = IM.open(path)
            boxes.append(image.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).cuda()
        save_image(templates, f"{output_dir}/results/templates.png", nrow=7)
        self.ref_feats = self.model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    )
        logging.info(f"Ref feats: {self.ref_feats.shape}")

        rospy.init_node("cnos_custom_detection")
        self.server = actionlib.SimpleActionServer('/object_detector/cnos_custom',
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

        start_time = time.time()
        detections = self.model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        decriptors = self.model.descriptor_model.forward(np.array(rgb), detections)
        logging.info(f"Time: {time.time() - start_time}")


        # get scores per proposal
        scores = self.metric(decriptors[:, None, :], self.ref_feats[None, :, :])
        score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
        score_per_detection = torch.mean(
            score_per_detection, dim=-1
        )
        
        # get top-k detections
        scores, index = torch.topk(score_per_detection, k=self.num_max_dets, dim=-1)
        detections.filter(index)
        
        # keep only detections with score > conf_threshold
        detections.filter(scores>self.conf_threshold)
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))
            
        detections.to_numpy()

        response = detections.return_results_dict(
            runtime=2,
            dataset_name="test",
        )

        response = response
        category_id = response['category_id']
        scores = response['score']
        bbox = response['bbox']
        mask = response['segmentation']

        print(scores)
        if len(bbox) == 0:
            logging.info(f"No object with conficence > {self.conf_threshold} detected")
            self.server.set_aborted()
            return

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
        result.class_confidences = scores[0:i+1]
        result.image = ros_numpy.msgify(Image, label_image, encoding='16SC1')
        result.class_names = [item_dict[i] for i in category_id[0:i+1]]

        result.class_names = [self.object_name]
        print("\nDetected Objects:\n")
        print(result.class_names)
        print(result.class_confidences)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        self.server.set_succeeded(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", default='custom_data/mesh.ply', help="Path to the model file")
    parser.add_argument(
        "obj_pose", 
        nargs="?", 
        default='src/poses/predefined_poses/obj_poses_level0.npy', 
        help="Path to the model file")
    parser.add_argument(
        "output_dir", nargs="?", default='custom_data/temps', help="Path to where the final files will be saved"
    )
    parser.add_argument("gpus_devices", nargs="?", default='0', help="GPU devices")
    parser.add_argument("light_itensity", nargs="?", type=float, default=0.1, help="Light itensity")
    parser.add_argument("radius", nargs="?", type=float, default=0.3, help="Distance from camera to object")
    parser.add_argument("--num_max_dets", nargs="?", default=1, type=int, help="Number of max detections")
    parser.add_argument("--confg_threshold", nargs="?", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    CNOS_ROS(args.stability_score_thresh, args.num_max_dets, args.confg_threshold, args.gpus_devices, args.cad_path, args.obj_pose, args.output_dir, args.light_itensity, args.radius)
    rospy.spin()