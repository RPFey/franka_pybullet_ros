import os
import pkgutil
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as sciR

from franka_env.panda_robot import FrankaPanda # type: ignore
import pybullet_utils.bullet_client as bc
import numpy as np
import importlib

dirname = os.path.dirname(__file__)
ycb_database = os.path.join(dirname, "../", "ycb_objects")

class FrankaPandaEnv:
    def __init__(self, connection_mode=p.GUI, frequency=1000., controller='position',
                 include_gripper=True, simple_model=False,
                 object_from_sdf=None, object_from_list=None, 
                 remove_box=False, seed=42):

        self.bc = bc.BulletClient(connection_mode)
        egl = pkgutil.get_loader('eglRenderer')
        self.rng = np.random.default_rng(seed=seed)
        self.bc.loadPlugin(egl.get_filename(), "eglRendererPlugin")
        self.bc.configureDebugVisualizer(rgbBackground=[33. / 255., 90. / 255., 127. / 255.])
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_GUI, 0)
        self.frequency = frequency
        self.remove_box = remove_box
        self.bc.setTimeStep(1 / frequency)
        self.bc.setGravity(0, 0, -9.81)
        
        self.bc.setRealTimeSimulation(0) 
        self.bc.setPhysicsEngineParameter(numSolverIterations=500, deterministicOverlappingPairs=1, enableFileCaching=0)

        self.plane_id = None
        self.table_id = None
        self.object_id = []
        self.object_list = ["YcbBanana", "YcbPear", "YcbHammer", "YcbScissors", "YcbStrawberry", "YcbChipsCan",
                            "YcbCrackerBox", "YcbFoamBrick", "YcbGelatinBox", "YcbMasterChefCan", "YcbMediumClamp", 
                            "YcbMustardBottle", "YcbPottedMeatCan", "YcbPowerDrill", "YcbTennisBall", "YcbTomatoSoupCan"]
        self.id2names = {}

        if object_from_list:
            self.add_ycb_objects_from_list(self.object_list)
        elif object_from_sdf:
            self.add_ycb_objects_from_sdf(object_from_sdf)

        self.controller = controller
        self.include_gripper = include_gripper
        self.simple_model = simple_model
        self.object_from_list = object_from_list
        self.object_from_sdf = object_from_sdf
        self.panda_robot = FrankaPanda(self.bc, include_gripper=include_gripper, simple_model=simple_model)
        
        
        # camera parameters
        self.camera_width = 800
        self.camera_height = 800
        self.camera_fovx = np.pi / 2
        self.camera_fovy = np.pi / 2
        self.focal_x = (self.camera_width / 2) / (np.tan(self.camera_fovx / 2))
        self.focal_y = (self.camera_height / 2) / (np.tan(self.camera_fovy / 2))
        self.camera_near = 0.02
        self.camera_far = 5.00
        self.K = np.array([[self.focal_x, 0., self.camera_width / 2], [0., self.focal_y, self.camera_height / 2], [0., 0., 1.]])
        self.projection_matrix = np.array([
            [2 / self.camera_width * self.K[0, 0], 0, (self.camera_width - 2 * self.K[0, 2]) / self.camera_width, 0],
            [0, 2 / self.camera_height * self.K[1, 1], (2 * self.K[1, 2] - self.camera_height) / self.camera_height, 0],
            [0, 0, (self.camera_near + self.camera_far) / (self.camera_near - self.camera_far),
             2 * self.camera_near * self.camera_far / (self.camera_near - self.camera_far)],
            [0, 0, -1, 0]]).T

    def simulate_step(self):
        self.bc.stepSimulation()

    def simulation_loop(self):
        import time
        while True:
            start = time.time()
            self.simulate_step()
            print(1 / (time.time() - start))
        
        self.bc.disconnect()
        print("Simulation end")

    def add_urdf_object(self, filename):
        flags = self.bc.URDF_USE_INERTIA_FROM_FILE
        noise_x = self.rng.uniform(-0.05, 0.05)
        noise_y = self.rng.uniform(-0.05, 0.05)
        self.object_id.append(self.bc.loadURDF(filename, [0.5 + noise_x, 0.0 + noise_y, 0.4], flags=flags))

    def add_ycb_objects_from_list(self, object_list, object_num=10):
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
                                         basePosition=[0, 0, -0.65], useFixedBase=True)
        self.table_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                         basePosition=[0.5, 0, -0.65], useFixedBase=True)
        self.box_id = self.bc.loadURDF(os.path.join(ycb_database, "OpenBox/box.urdf"),
                                        basePosition=[0.36, -0.15, -0.02], useFixedBase=True)
        self.bc.setAdditionalSearchPath(os.path.join(dirname, "grasp_sdf_env"))
        obj_idx = np.arange(len(object_list))
        self.rng.shuffle(obj_idx)
        # np.random.shuffle(obj_idx)
        obj_idx = obj_idx[:object_num]

        print("Loading objects from a list")
        for i in obj_idx:
            obj = object_list[i]
            self.bc.setAdditionalSearchPath(os.path.join(ycb_database,  obj))
            self.add_urdf_object(os.path.join(ycb_database,  obj,  "model.urdf"))
            self.id2names[self.object_id[-1]] = obj
            self.wait_for_objects_to_rest()
                
        if self.remove_box:
            self.bc.removeBody(self.box_id)
        self.wait_for_objects_to_rest(1e-3, max_step=6000)
        
        np.set_printoptions(precision=8, suppress=True)
        for i, body in zip(obj_idx, self.object_id):
            pos, quat = self.bc.getBasePositionAndOrientation(body)
            pos = np.array(pos)
            # euler = self.bc.getEulerFromQuaternion(quat)
            print(object_list[i], np.concatenate([pos, quat]))
        np.set_printoptions(precision=8, suppress=False)
        
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

    def add_ycb_objects_from_sdf(self, object_sdf):
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
                                         basePosition=[0, 0, -0.65], useFixedBase=True)
        self.table_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                         basePosition=[0.5, 0, -0.65], useFixedBase=True)

        print("Loading objects from a SDF")
        self.bc.setAdditionalSearchPath(os.path.join(dirname, "grasp_sdf_env"))
        self.bc.setAdditionalSearchPath(ycb_database)
        
        with open(object_sdf, 'r') as f:
            objects = f.readlines()
        
        object_id_temp = []
        for obj in objects:
            items = [k for k in obj.split(' ') if len(k) > 0]
            typename = items[0]

            poses = np.array([float(k) for k in items[1:]]).astype(np.float64)
            poses[2] += 0.02
            filename = os.path.join(ycb_database,  typename,  "model.urdf")

            flags = self.bc.URDF_USE_INERTIA_FROM_FILE
            object_id_temp.append(self.bc.loadURDF(filename, poses[:3], poses[3:], flags=flags))

        # object_id_temp = self.bc.loadSDF(object_sdf)

        self.object_id += object_id_temp
        self.wait_for_objects_to_rest()
        np.set_printoptions(precision=4, suppress=True)
        for body in object_id_temp:
            pos, quat = self.bc.getBasePositionAndOrientation(body)
            pos = np.array(pos)
            euler = self.bc.getEulerFromQuaternion(quat)
            # euler = sciR.from_quat(quat).as_euler('xyz')
            print(np.concatenate([pos, euler]))
        np.set_printoptions(precision=8, suppress=False)

        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)
    
    def remove_objects(self):
        self.bc.removeBody(self.plane_id)
        self.bc.removeBody(self.table_id)
        for obj in self.object_id:
            self.bc.removeBody(obj)
        self.object_id = []
    
    def reset(self):
        # Reset the simulation
        # self.bc.resetSimulation()
        self.bc.setGravity(0, 0, -9.81)
        self.bc.setTimeStep(1 / self.frequency)

        self.panda_robot.reset_state()
        self.panda_robot.open_gripper()
        
        self.remove_objects()
        
        # Reload objects
        self.object_id = []
        if self.object_from_list:
            self.add_ycb_objects_from_list(self.object_list)
        elif self.object_from_sdf:
            self.add_ycb_objects_from_sdf(self.object_from_sdf)
            
    def wait_for_objects_to_rest(self, tol=0.01, max_step=200):
        step = 0
        
        print("Current Objects: ", len(self.object_id))
        objects_resting = False
        while not objects_resting and step < max_step:
            # simulate a quarter of a second
            for _ in range(100):
                self.bc.stepSimulation()
            
            # check whether all objects are resting
            objects_resting = True
            for body in self.object_id:
                vel = self.bc.getBaseVelocity(body)
                if np.linalg.norm(vel) > tol:
                    objects_resting = False
                    break
            
            step += 1
