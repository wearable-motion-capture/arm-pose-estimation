import logging
import time

import numpy as np

from nat_net.NatNetClient import NatNetClient
from utility.messaging import motive_bone_ids



frame_count = 0

# stores position and rotation of bones of interest
bones = {
    motive_bone_ids["Hips"]: [np.zeros(0), np.zeros(4)],
    motive_bone_ids["Spine"]: [np.zeros(0), np.zeros(4)],
    motive_bone_ids["Chest"]: [np.zeros(0), np.zeros(4)],
    motive_bone_ids["LeftShoulder"]: [np.zeros(0), np.zeros(4)],
    motive_bone_ids["LeftUpperArm"]: [np.zeros(0), np.zeros(4)],
    motive_bone_ids["LeftLowerArm"]: [np.zeros(0), np.zeros(4)],
    motive_bone_ids["LeftHand"]: [np.zeros(0), np.zeros(4)]
}



def receive_new_frame(data_dict):
    """
    simply used for Hz estimations
    :param data_dict:
    :return:
    """
    frame_count = frame_count + 1


def receive_rigid_body_frame(new_id, position, rotation):
    """
    This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    :param new_id:
    :param position:
    :param rotation:
    :return:
    """
    if new_id in bones:
        bones[new_id] = [np.array(position), np.array(rotation)]


if __name__ == "__main__":

    # start ros node
    logging.basicConfig(level=logging.INFO)

    streaming_client = NatNetClient()

    client_ip = "192.168.1.138"  # this machine
    server_ip = "192.168.1.116"  # motive machine

    streaming_client.set_client_address(client_ip)
    streaming_client.set_server_address(server_ip)
    streaming_client.set_use_multicast(False)  # only works in unicast setting

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame
    streaming_client.set_print_level(0)  # Showing only received frame numbers and supressing data descriptions

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    logging.info("Start motive streaming client")
    try:
        is_running = streaming_client.run()

        if not is_running:
            raise UserWarning("ERROR: Could not start streaming client.")

        is_looping = True
        time.sleep(1)
        if streaming_client.connected() is False:
            raise UserWarning("ERROR: Could not connect properly.  Check that Motive streaming is on.")

        # loop forever until an error occurs
        while is_looping:
            time.sleep(1)
    finally:
        streaming_client.shutdown()
