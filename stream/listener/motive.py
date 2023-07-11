import logging
import time

import numpy as np

import utility.transformations as ts
import config
from nat_net.MoCapData import MoCapData
from nat_net.NatNetClient import NatNetClient
from utility.messaging import motive_bone_ids


class MotiveListener:
    def __init__(self):

        self.__tag = "MOTIVE LISTENER"
        self.__frame_count = 0  # for Hz estimations

        # stores position and rotation of bones of interest
        self.__bones = {}

        self.__smallest_id = None  # motive sets the skeleton body IDs incrementally. We reduce them to start with 1

        self.__streaming_client = NatNetClient()

        self.__streaming_client.set_client_address(config.IP)  # this machine
        self.__streaming_client.set_server_address(config.MOTIVE_SERVER)  # motive machine
        self.__streaming_client.set_use_multicast(False)  # only works in unicast setting

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        self.__streaming_client.new_frame_listener = self.receive_new_frame
        self.__streaming_client.set_print_level(
            0)  # Showing only received frame numbers and suppressing data descriptions

    def stream_loop(self):
        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        logging.info("[{}] Start motive streaming client".format(self.__tag))
        try:
            is_running = self.__streaming_client.run()

            if not is_running:
                raise UserWarning("ERROR: Could not start streaming client.")

            is_looping = True
            time.sleep(1)
            if self.__streaming_client.connected() is False:
                raise UserWarning("ERROR: Could not connect properly.  Check that Motive streaming is on.")

            # loop forever until an error occurs
            while is_looping:
                time.sleep(5)
                logging.info("[{}] {} Hz".format(self.__tag, self.__frame_count / 5))
                self.__frame_count = 0
        finally:
            self.__streaming_client.shutdown()

    def receive_new_frame(self, data_dict, mocap_data: MoCapData):
        """
        called when receiving a new frame. Overwrites the current frame and increases count for Hz estimations
        :param data_dict:
        :return:
        """
        self.__frame_count += 1
        # write bones into new dictionary and replace old one
        new_bones = {}
        rb_dat = mocap_data.rigid_body_data
        # find the smallest ID
        if self.__smallest_id is None:
            self.__smallest_id = min([x.id_num for x in rb_dat.rigid_body_list]) - 1
        for rb in rb_dat.rigid_body_list:
            if rb.tracking_valid:
                new_bones[rb.id_num - self.__smallest_id] = [np.array(rb.pos), np.array(rb.rot)]
        self.__bones = new_bones

    def get_ground_truth(self):
        # snapshot of current bone positions
        cb = self.__bones

        try:
            # limb rotations of interest
            hip_rot_g = ts.mocap_quat_to_global(cb[motive_bone_ids["Hips"]][1])
            hand_rot_g = ts.mocap_quat_to_global(cb[motive_bone_ids["LeftHand"]][1])
            uarm_rot_g = ts.mocap_quat_to_global(cb[motive_bone_ids["LeftUpperArm"]][1])
            larm_rot_g = ts.mocap_quat_to_global(cb[motive_bone_ids["LeftLowerArm"]][1])

            # limb origins of interest
            uarm_orig_g = ts.mocap_pos_to_global(cb[motive_bone_ids["LeftUpperArm"]][0])
            larm_orig_g = ts.mocap_pos_to_global(cb[motive_bone_ids["LeftLowerArm"]][0])
            hand_orig_g = ts.mocap_pos_to_global(cb[motive_bone_ids["LeftHand"]][0])
        except KeyError:
            self.__smallest_id = None  # IDs could be wrong because of ID change
            return None

        # estimate rotations relative to hip
        hand_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), hand_rot_g)
        larm_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), larm_rot_g)
        uarm_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), uarm_rot_g)

        # estimate positions relative to upper arm origin
        larm_origin_rua = ts.quat_rotate_vector(ts.quat_invert(hip_rot_g), np.array(larm_orig_g - uarm_orig_g))
        hand_orig_rua = ts.quat_rotate_vector(ts.quat_invert(hip_rot_g), np.array(hand_orig_g - uarm_orig_g))

        return np.hstack([hand_rot_rh, hand_orig_rua, larm_rot_rh, larm_origin_rua, uarm_rot_rh])

    @staticmethod
    def get_ground_truth_header():
        """
        descriptive labels for what get_ground_truth() returns
        :return:
        """
        return [
            # hand rotation
            "hand_rot_rh_w", "hand_rot_rh_x", "hand_rot_rh_y", "hand_rot_rh_z",
            # hand position
            "hand_orig_rua_x", "hand_orig_rua_y", "hand_orig_rua_z",
            # larm rotation
            "larm_rot_rh_w", "larm_rot_rh_x", "larm_rot_rh_y", "larm_rot_rh_z",
            # elbow position (larm origin)
            "larm_orig_rua_x", "larm_orig_rua_y", "larm_orig_rua_z",
            # uarm rotation
            "uarm_rot_rh_w", "uarm_rot_rh_x", "uarm_rot_rh_y", "uarm_rot_rh_z",
        ]
