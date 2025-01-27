import multiprocessing as mp
import time
import cv2
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as sciR
from franka_env.env import FrankaPandaEnv

def quaternion_from_matrix(R, strict_check=True):
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0.0:
        sqrt_trace = np.sqrt(1.0 + trace)
        q[3] = 0.5 * sqrt_trace
        q[0] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
        q[1] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
        q[2] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            sqrt_trace = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
            q[0] = 0.5 * sqrt_trace
            q[1] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[2] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
        elif R[1, 1] > R[2, 2]:
            sqrt_trace = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
            q[0] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[1] = 0.5 * sqrt_trace
            q[2] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
        else:
            sqrt_trace = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
            q[0] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
            q[1] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
            q[2] = 0.5 * sqrt_trace
    return q


class FrankaPandaEnvPhysics(FrankaPandaEnv):
    def __init__(self, connection_mode=p.GUI, frequency=1000., controller='position',
                 include_gripper=True, simple_model=False, 
                 object_from_sdf=None, object_from_list=None, remove_box=False):
        
        super().__init__(connection_mode=connection_mode, frequency=frequency, controller=controller,
                         include_gripper=include_gripper, simple_model=simple_model,
                         object_from_sdf=object_from_sdf, object_from_list=object_from_list, remove_box=remove_box)

        if self.controller == 'position':
            self.current_joint_input = self.panda_robot.home_joint[:self.panda_robot.dof]
        else:
            self.current_joint_input = [0, 0, 0, 0, 0, 0, 0]
        self.current_gripper_input = True
        self.panda_robot.open_gripper()
        self.video_step = 0 
        
    def get_hand_eye(self):
        """
        Get the position and orientation of the camera and the hand
        :return: pos, orn, cam_pos, cam_orn
        """
        # compute camera
        camera_joint_id = 7
        joint_info = self.bc.getJointInfo(self.panda_robot.robot_id, camera_joint_id)
        parent_link = joint_info[0]
        link_info = self.bc.getLinkState(self.panda_robot.robot_id, parent_link)
        link_pose_world = link_info[0]
        link_ori_world = link_info[1]
        pos = np.array(link_pose_world)
        orn = link_ori_world
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))
        ee_T = np.eye(4)
        ee_T[:3, :3] = R
        ee_T[:3, 3] = pos
        ee_cam_T = np.array([[0.7061203, -0.7078083, -0.0200362, 0.00291476],
                                  [0.7080078, 0.7061898, 0.0045781, -0.04739189],
                                  [0.0109089, -0.0174185, 0.9997888, 0.0643254],
                                  [0., 0., 0., 1.]])
        cam_T = ee_T @ ee_cam_T
        cam_pos = cam_T[:3, 3]
        cam_orn = quaternion_from_matrix(cam_T[:3,:3])
        
        # compute hand
        hand_joint_id = 8
        joint_info = self.bc.getJointInfo(self.panda_robot.robot_id, hand_joint_id)
        parent_link = joint_info[0]
        link_info = self.bc.getLinkState(self.panda_robot.robot_id, parent_link)
        link_pose_world = link_info[0]
        link_ori_world = link_info[1]
        pos = np.array(link_pose_world)
        orn = link_ori_world
        
        return pos, orn, cam_pos, cam_orn

    def simulate_step(self):
        
        self.video_step = (self.video_step + 1) % 100
            
        if self.controller == 'position':
            self.panda_robot.set_target_positions(self.current_joint_input)
        elif self.controller == 'velocity':
            self.panda_robot.set_target_velocities(self.current_joint_input)
        elif self.controller == 'torque':
            self.panda_robot.set_target_torques(self.current_joint_input)
            
        self.bc.stepSimulation()
        
    def get_image(self, cam_pos, cam_orn):
        self.view_matrix = cvPose2BulletView(cam_pos, cam_orn)
        img = self.bc.getCameraImage(self.camera_width, self.camera_height, self.view_matrix,
                                     self.projection_matrix.reshape(-1).tolist(),
                                     renderer=self.bc.ER_BULLET_HARDWARE_OPENGL,
                                     flags=self.bc.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        color = cv2.cvtColor(np.array(img[2]), cv2.COLOR_RGB2BGR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = self.camera_far * self.camera_near / (
                self.camera_far - (self.camera_far - self.camera_near) * np.array(img[3]))
        
        # depth = np.where(depth >= self.camera_far, np.zeros_like(depth), depth)
        return color, depth
    
def run_simulation(mode, object_from_sdf, object_from_list,
                        joint_input, joint_data, 
                            gripper, camera_pose, ee_pose, 
                                image_rgb, image_depth, stop):
    
    frequency = 1000.
    env = FrankaPandaEnvPhysics(connection_mode=mode,
                                frequency=frequency,
                                controller='position',
                                include_gripper=True,
                                simple_model=True,
                                object_from_sdf=object_from_sdf,
                                object_from_list=object_from_list)
    
    # get home joints
    joint_input_np_array = np.frombuffer(joint_input.get_obj(), dtype=np.float32)
    joint_input_np_array[:env.panda_robot.dof] = env.panda_robot.home_joint[:env.panda_robot.dof]
    # set to none
    gripper.value = 0
        
    while stop.value == 0:
        with joint_input.get_lock():
            joint_input_np_array = np.frombuffer(joint_input.get_obj(), dtype=np.float32)
            
        env.current_joint_input[:] = np.array(joint_input)
        
        if gripper.value == 0:
            env.panda_robot.open_gripper()
        else:
            env.panda_robot.close_gripper()
        
        env.simulate_step()
        
        # write joint state
        pos_reading, vel_reading, force_reading, effort_reading = env.panda_robot.get_pos_vel_force_torque()
        with joint_data.get_lock():
            joint_data_np_array = np.frombuffer(joint_data.get_obj(), dtype=np.float32).reshape((3, 9))
            joint_data_np_array[:, :len(env.panda_robot.joints)] = np.stack([pos_reading, vel_reading, effort_reading])
        
        # write camera data
        if env.video_step == 0: 
            hand_pos, hand_orn, cam_pos, cam_orn = env.get_hand_eye()  
            color, depth = env.get_image(cam_pos, cam_orn)
            
            with image_rgb.get_lock():
                image_rgb_np_array = np.frombuffer(image_rgb.get_obj(), dtype=np.int32).reshape((800, 800, 3))
                image_rgb_np_array[:, :, :] = color.reshape((800, 800, 3))
                
            with image_depth.get_lock():
                image_depth_np_array = np.frombuffer(image_depth.get_obj(), dtype=np.float32).reshape((800, 800))
                image_depth_np_array[:, :] = depth.reshape((800, 800))
        
        # write ee pose and camera pos
        hand_pos, hand_orn, cam_pos, cam_orn = env.get_hand_eye()
        with camera_pose.get_lock():
            camera_pose_np_array = np.frombuffer(camera_pose.get_obj(), dtype=np.float32).reshape((4, 4))
            orn = sciR.from_quat(cam_orn)
            camera_pose_np_array[:3, 3] = cam_pos
            camera_pose_np_array[:3, :3] = orn.as_matrix()
            camera_pose_np_array[3, :3] = 0.
            camera_pose_np_array[3, 3] = 1.
        
        with ee_pose.get_lock():
            ee_pose_np_array = np.frombuffer(ee_pose.get_obj(), dtype=np.float32).reshape((4, 4))
            orn = sciR.from_quat(hand_orn)
            ee_pose_np_array[:3, 3] = hand_pos
            ee_pose_np_array[:3, :3] = orn.as_matrix()
            ee_pose_np_array[3, :3] = 0.
            ee_pose_np_array[3, 3] = 1.
            
        time.sleep(1 / frequency)
            
    env.bc.disconnect()
    print("Simulation end")
        
def cvPose2BulletView(t, q):
    """
    cvPose2BulletView gets orientation and position as used
    in ROS-TF and opencv and coverts it to the view matrix used
    in openGL and pyBullet.

    :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw]
    :param t: ROS postion expressed as [tx, ty, tz]
    :return:  4x4 view matrix as used in pybullet and openGL

    """
    R = np.array(p.getMatrixFromQuaternion(q)).reshape((3, 3))

    T = np.vstack([np.hstack([R, np.array(t).reshape(3, 1)]),
                   np.array([0, 0, 0, 1])])
    # Convert opencv convention to python convention
    # By a 180 degrees rotation along X
    Tc = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]]).reshape(4, 4)

    # pybullet pse is the inverse of the pose from the ROS-TF
    T = Tc @ np.linalg.inv(T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = T.T.reshape(16)
    return viewMatrix

class FrankaClutter:
    def __init__(self, object_from_sdf=None, 
                        object_from_list=True, gui=False):
        mp.set_start_method('spawn')
    
        image_width = 800 # self._env.camera_width
        image_height = 800 # self._env.camera_height
        self.intrinsic = np.array([[400.0, 0.0, 400.],
                                    [0.0, 400.0, 400.],
                                    [0.0, 0.0, 1.0]])
        mode = p.GUI if gui else p.DIRECT
    
        self._joint_input = mp.Array('f', 7)
        self._joint_data = mp.Array('f', 3 * 9)
        self._gripper = mp.Value('i', 1)
        self._camera_pose = mp.Array('f', 16)
        self._ee_pose = mp.Array('f', 16)
        self._image_rgb = mp.Array('i', image_width * image_height * 3)
        self._image_depth = mp.Array('f', image_width * image_height)
        self._stop = mp.Value('i', 0)
        
        print("Start Env ...")
        self._process = mp.Process(target=run_simulation, args=(mode, object_from_sdf, object_from_list, 
                                                                self._joint_input, self._joint_data, 
                                                                self._gripper, self._camera_pose, self._ee_pose, 
                                                                self._image_rgb, self._image_depth, self._stop))
        self._process.start()
        
    def get_camera_intrinsic(self):
        return self.intrinsic.copy()
    
    def get_camera_pose(self):
        with self._camera_pose.get_lock():
            camera_pose = np.frombuffer(self._camera_pose.get_obj(), dtype=np.float32).reshape((4, 4))
        return camera_pose.copy()
        
    def get_ee_pose(self):
        with self._ee_pose.get_lock():
            ee_pose = np.frombuffer(self._ee_pose.get_obj(), dtype=np.float32).reshape((4, 4))
        return ee_pose.copy()
        
    def get_image(self):
        image = np.frombuffer(self._image_rgb.get_obj(), dtype=np.int32).reshape((800, 800, 3))
        depth = np.frombuffer(self._image_depth.get_obj(), dtype=np.float32).reshape((800, 800))
        return image.copy().astype(np.uint8), depth.copy()
    
    def get_joint_data(self):
        """  
        Returns:
            np.array: joint data of the robot 3 x 9
                position, velocity, force
        """
        with self._joint_data.get_lock():
            joint_data = np.frombuffer(self._joint_data.get_obj(), dtype=np.float32).reshape(3, 9)
            
        return joint_data.copy()
    
    def close_gripper(self):
        with self._gripper.get_lock():
            self._gripper.value = 1
        
    def open_gripper(self):
        with self._gripper.get_lock():
            self._gripper.value = 0
        
    def set_joint_input(self, joint_input):
        with self._joint_input.get_lock():
            shared_np_array = np.frombuffer(self._joint_input.get_obj(), dtype=np.float32)
            shared_np_array[:] = joint_input
    
    def end(self):
        self._stop.value = 1
        self._process.join()
        

if __name__ == "__main__":
    env = FrankaClutter() 
    try:
        while True:
            rgb, depth = env.get_image()
            if rgb.max () > 10:
                # cv2.imshow("rgb", rgb)
                # cv2.waitKey(1)
                # raise KeyboardInterrupt()
                env.close_gripper()
                print("Gripper closed")
                break
            else:
                print("Wait for init ...")
                time.sleep(1)
        
    except KeyboardInterrupt:
        env.end()
        print("Simulation end")