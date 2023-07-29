import os

import numpy as np
import pandas as pd

import wear_mocap_ape.config as config
from wear_mocap_ape.data_types.bone import Bone
import xml.etree.ElementTree as ElTr

# mapping from XML IDs to Unity Bone IDs. IDs and Names are the same as in DollAnimator of the Unity project
XML_TO_MECANIM = {
    1: "Hips",
    2: "Spine",
    3: "Chest",
    4: "Neck",
    5: "Head",
    6: "LeftShoulder",
    7: "LeftUpperArm",
    8: "LeftLowerArm",
    9: "LeftHand",
    10: "RightShoulder",
    11: "RightUpperArm",
    12: "RightLowerArm",
    13: "RightHand",
    14: "LeftUpperLeg",
    15: "LeftLowerLeg",
    16: "LeftFoot",
    17: "LeftToes",
    18: "RightUpperLeg",
    19: "RightLowerLeg",
    20: "RightFoot",
    21: "RightToes"
}


class BoneMap:
    """
    parses a bone map with distances in T-pose from a mocap skeleton.xml
    """

    # available without initialization
    DEFAULT_LARM_LEN = 0.22
    DEFAULT_UARM_LEN = 0.30

    def __init__(self, skeleton: str):

        self.__skeleton_name = None
        self.__bonemap = {}  # internal structure is a dict

        # location of skeleton folder (where all XML files are stored)
        skel_path = os.path.join(config.paths["skeleton"], "{}.xml".format(skeleton))
        tree = ElTr.parse(skel_path)

        # traverse XML hierarchy and parse into dict of Bones
        root = tree.getroot()
        bones = root.find("NodeAssets").find("skeleton").find("bones")
        # parse bone entries into bonemap dict
        for bone in bones.findall('bone'):
            b_id = int(bone.get("id"))
            p_id = int(bone.find("parent_id").text)
            # parse bone offsets from XML
            b_pos = np.array([float(x) for x in bone.find("offset").text.split(",")], dtype=np.float64)
            # if the bone is not the root object, add parent bone transform for global position
            b_pos += np.zeros(3) if p_id == 0 else self.__bonemap[p_id].default_pos
            # create bone object and add to bone map
            self.__bonemap[b_id] = Bone(bone_id=b_id, default_pos=b_pos)

        # traverse XML hierarchy to find the skeleton name property
        root = tree.getroot()
        props = root.find("NodeAssets").find("skeleton").find("properties")
        for prop in props.findall('property'):
            if prop.find("name").text == "NodeName":
                self.__skeleton_name = prop.find("value").text

    def create_default_t_pose_df(self, time_steps_s: pd.Series):
        """
        create a prediction data frame with default values. One row for every time step of the sw_ext_data
        :param time_steps_s: time steps of observations in seconds
        :return:
        """
        def_dict = {"time_s": time_steps_s}
        for k, v in self.__bonemap.items():
            name = XML_TO_MECANIM[k]  # map the name for unique column names
            # add rotation entries
            for rot, descr in zip(v.default_rot, ["_rot_w", "_rot_x", "_rot_y", "_rot_z"]):
                def_dict.update({name + descr: rot})
            # add origin entries
            for origin, descr in zip(v.default_pos, ["_pos_x", "_pos_y", "_pos_z"]):
                def_dict.update({name + descr: origin})
        # now we have a (timesteps x bone_values) data frame filled with default t-pose bone positions
        return pd.DataFrame(def_dict, index=range(len(time_steps_s)))

    @property
    def skeleton_name(self):
        if self.__skeleton_name is None:
            raise UserWarning("No skeleton name in bone map. Parsing error?")
        return self.__skeleton_name

    @property
    def left_upper_arm_origin_rh(self):
        """default left upper arm origin relative to hip (rh)"""
        return self.__bonemap[7].default_pos - \
            self.__bonemap[1].default_pos  # Hips

    @property
    def left_lower_arm_vec(self):
        """vector from left lower arm to hand (leftHand - leftLowerArm)"""
        return self.__bonemap[9].default_pos - self.__bonemap[8].default_pos

    @property
    def right_lower_arm_vec(self):
        """vector from right lower arm to hand (rightHand - rightLowerArm)"""
        return self.__bonemap[13].default_pos - self.__bonemap[12].default_pos

    @property
    def left_lower_arm_length(self):
        return np.linalg.norm(self.left_lower_arm_vec)

    @property
    def left_lower_arm_origin_g(self):
        return self.__bonemap[8].default_pos  # leftLowerArm

    @property
    def left_upper_arm_vec(self):
        """vector from left upper arm to elbow (leftLowerArm - leftUpperArm)"""
        return self.__bonemap[8].default_pos - self.__bonemap[7].default_pos

    @property
    def right_upper_arm_vec(self):
        """vector from right upper arm to elbow (rightLowerArm - rightUpperArm)"""
        return self.__bonemap[12].default_pos - self.__bonemap[11].default_pos

    @property
    def left_upper_arm_length(self):
        return np.linalg.norm(self.left_upper_arm_vec)

    @property
    def left_upper_arm_origin_g(self):
        return self.__bonemap[7].default_pos  # LeftUpperArm

    @property
    def hip_origin_g(self):
        return self.__bonemap[1].default_pos  # hips
