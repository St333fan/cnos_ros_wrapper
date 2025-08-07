import numpy as np
import time
from PIL import Image as IM
import logging
# set level logging
logging.basicConfig(level=logging.INFO)
import argparse
import rospy
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
#import torch transforms
import ros_numpy
from sensor_msgs.msg import Image
from CNOS import CNOSDetector


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

#item_dict = {
#    1: '006_mustard_bottle',
#    2: '024_bowl',
#}

class CNOS_ROS:
    def __init__(self, templates_dir, stability_score_thresh, conf_threshold, subset, item_dict=item_dict):
        print(f"Initializing CNOS Object Detector with params: {templates_dir=}, {stability_score_thresh=}, {conf_threshold=}, {subset=}")
        self.cnos_detector = CNOSDetector(
            templates_dir = templates_dir,
            conf_threshold=conf_threshold, 
            stability_score_thresh=stability_score_thresh, 
            config_name="run_inference.yaml",
            subset=subset
        )
        self.item_dict = item_dict

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

        # numpy to PIL
        rgb = IM.fromarray(rgb)

        start_time = time.time()

        results = self.cnos_detector.run_inference(rgb)
        #  dict with obj_ids masks and scores

        category_id = results['obj_ids']
        scores = results['scores']
        masks = results['masks']

        print(f"Detection Scores: {scores}")
        if len(masks) == 0:
            rospy.loginfo(f"No object with conficence > {self.conf_threshold} detected")
            self.server.set_aborted()
            return

        # sort the four np.arrays based on the scores
        order = np.argsort(scores)[::-1]
        category_id = category_id[order]
        scores = scores[order]
        masks = masks[order]
        label_image = np.full_like(masks[0],-1, dtype=np.int16)
        for i, score in enumerate(scores):
            label_image[masks[i]>0] = i

        result = GenericImgProcAnnotatorResult()
        result.success = True
        result.class_confidences = scores[0:i+1]
        result.image = ros_numpy.msgify(Image, label_image, encoding='16SC1')
        result.class_names = [self.item_dict[i+1] for i in category_id[0:i+1]]

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
    parser.add_argument("--templates_dir", required=True, type=str, help="Path to the templates folder")
    parser.add_argument("--confg_threshold", nargs="?", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.97, type=float, help="stability_score_thresh of SAM")
    parser.add_argument("--subset", nargs="?", default=4, type=int, help="uses every nth template")
    args = parser.parse_args()
    CNOS_ROS(args.templates_dir, args.stability_score_thresh, args.confg_threshold, args.subset)
    rospy.spin()
