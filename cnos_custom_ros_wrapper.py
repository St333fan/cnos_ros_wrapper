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

item_dict = {
    1: '001_ahorn_sirup',
    2: '002_max_house',
}

item_dict = {
    1: '006_mustard_bottle', 
    2: '010_potted_meat_can',
    3: '024_bowl',
    4: '003_cracker_box',
    5: '002_master_chef_can',
    6: '009_gelatin_box',
    7: '040_large_marker',
    8: '052_extra_large_clamp',
}

item_dict = {    
    1: "101_soap",
    2: "102_ahorn_sirup",
    3: "103_tomato_paste",
    4: "104_kokos_can",
    5: "105_hand_cream",
    6: "106_wet_wipes",
    7: "107_razors",
    8: "108_balsamic",
    9: "109_toothbrush"
}

item_dict = {
    1: '006_mustard_bottle', 
    2: '010_potted_meat_can',
    3: "101_soap",
    4: "102_ahorn_sirup",
    5: "105_hand_cream",
    6: '009_gelatin_box',
    7: "107_razors",
    8: "109_toothbrush"
}

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
        self.server = actionlib.SimpleActionServer('/object_detector/sam6dism',
                                                    GenericImgProcAnnotatorAction,
                                                    execute_cb=self.detect_objects,
                                                    auto_start=False) # /object_detector/cnos_custom'
        self.server.start()

        print("Object detection with CNOS is ready.")


    """
    When using the robokudo_msgs, as the callback function for the action server
    Only send the best (highest confidence) object per category.
    """
    def detect_objects(self, goal):
        print("Detecting Objects\n")
        
        start_time = time.time()
        rgb = goal.rgb

        rgb = ros_numpy.numpify(rgb)

        # numpy to PIL
        rgb = IM.fromarray(rgb)

        results = self.cnos_detector.run_inference(rgb)
        category_id = results['obj_ids']
        scores = results['scores']
        masks = results['masks']

        print(f"Detection Scores: {scores}")
        if len(masks) == 0:
            rospy.loginfo(f"No object with confidence > {self.cnos_detector.conf_threshold} detected")
            self.server.set_aborted()
            return

        # Find the best (highest score) detection per category
        best_indices = {}
        for idx, cat in enumerate(category_id):
            if cat not in best_indices or scores[idx] > scores[best_indices[cat]]:
                best_indices[cat] = idx

        # Sort by score descending
        selected = sorted(best_indices.values(), key=lambda i: scores[i], reverse=True)

        # Prepare label image
        label_image = np.full_like(masks[0], -1, dtype=np.int16)
        for label, idx in enumerate(selected):
            label_image[masks[idx] > 0] = label

        result = GenericImgProcAnnotatorResult()
        result.success = True
        result.class_confidences = [scores[idx] for idx in selected]
        result.image = ros_numpy.msgify(Image, label_image, encoding='16SC1')
        result.class_names = [self.item_dict[category_id[idx] + 1] for idx in selected]

        print("\nDetected Objects (best per category):\n")
        print(result.class_names)
        print(result.class_confidences)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        self.server.set_succeeded(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates_dir", required=True, default="/code/datasets/templates_pyrender", type=str, help="Path to the templates folder")
    parser.add_argument("--conf_threshold", nargs="?", default=0.30, type=float, help="Confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.97, type=float, help="stability_score_thresh of SAM")
    parser.add_argument("--subset", nargs="?", default=16, type=int, help="uses every nth template")
    args = parser.parse_args()
    CNOS_ROS(args.templates_dir, args.stability_score_thresh, args.conf_threshold, args.subset)
    rospy.spin()
