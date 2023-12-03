import numpy as np


class Bone:
    """
    helper class to manage parsed bone objects from Motive XML.
    Used in BoneMap
    """

    def __init__(self,
                 bone_id: int,
                 default_pos,
                 default_rot=np.array([1, 0, 0, 0])  # identity quaternion w,x,y,z
                 ):
        self.bone_id = bone_id  # int
        self.default_pos = default_pos  # position as vec3
        self.default_rot = default_rot  # rotation as quaternion
