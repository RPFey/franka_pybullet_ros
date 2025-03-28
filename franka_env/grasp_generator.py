from pathlib import Path
import time
import os

import numpy as np
import pybullet as p 
import pybullet_data
import open3d as o3d
import enum
import franka_env.btsim as btsim
from franka_env.btsim import Rotation, Transform, CameraIntrinsic
from pathlib import Path as Path

class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed

dirname = os.path.dirname(__file__)
ycb_database = os.path.join(dirname, "../", "ycb_objects")

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width

def workspace_lines(size):
    return [
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
        [0.0, size, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, size],
        [size, 0.0, size],
        [size, size, size],
        [size, size, size],
        [0.0, size, size],
        [0.0, size, size],
        [0.0, 0.0, size],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, 0.0],
        [size, 0.0, size],
        [size, size, 0.0],
        [size, size, size],
        [0.0, size, 0.0],
        [0.0, size, size],
    ]

class ClutterRemovalSim(object):
    def __init__(self, scene, gui=True, seed=None):
        assert scene in ["pile", "packed"]

        self.urdf_root = ycb_database
        self.scene = scene
        self.discover_objects()

        self.global_scaling = 1. # {"blocks": 1.67}.get(object_set, 1.0)
        self.gui = gui

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        self.gripper = Gripper(self.world)
        self.size = 12 * self.gripper.finger_depth
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        """
            Find all objects under folder
            - root_data
                |
                - YcbXXXX
                    |
                    - XXXX.urdf
        """
        # iterate over all subfolders of self.urdf_root and find files with suffix .urdf
        folders = [f for f in Path(self.urdf_root).iterdir() if f.is_dir()]
        self.object_urdfs = []
        for folder in folders:
            if folder.name[:3] == 'Ycb':
                urdfs = list(folder.glob("*.urdf"))
                self.object_urdfs.extend(urdfs)
                
    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )
        
        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        plane = self.world.load_urdf(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"), pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(os.path.join(ycb_database, "OpenBox/box.urdf"), pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

    def acquire_tsdf(self, n, N=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
        r = 2.0 * self.size
        theta = np.pi / 6.0

        N = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        timing = 0.0
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)

        return tsdf, high_res_tsdf.get_cloud(), timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = os.path.join(ycb_database, "panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            p.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            p.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body, 0):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
    
def camera_on_sphere(origin, radius, theta, phi):
    from math import sin, cos
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()
    
def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    segmentations = np.empty((n, height, width), np.int32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        _, depth_img, mask = sim.camera.render(extrinsic) #[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        segmentations[i] = mask

    return depth_imgs, extrinsics, segmentations

def evaluate_grasp_point(sim, pose, num_rotations=6):
    pos = pose[:3, 3]
    # define initial grasp frame on object surface
    # z_axis = -normal
    # x_axis = np.r_[1.0, 0.0, 0.0]
    # if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
    #     x_axis = np.r_[0.0, 1.0, 0.0]
    # y_axis = np.cross(z_axis, x_axis)
    # x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(pose[:3, :3])

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        outcomes.append(outcome)
        widths.append(width)
        
    return outcomes, widths

def evaluate_grasp_pose(sim, pose):
    pos = pose[:3, 3]
    R = Rotation.from_matrix(pose[:3, :3])
    candidate = Grasp(Transform(R, pos), width=sim.gripper.max_opening_width)
    outcome, width = sim.execute_grasp(candidate, remove=False)
    return outcome, width

def main():
    sim = ClutterRemovalSim("pile", gui=True)
    sim.reset(2)
    sim.save_state()
    
    # render synthetic depth images
    MAX_VIEWPOINT_COUNT = 4
    n = np.random.randint(MAX_VIEWPOINT_COUNT) + 1
    depth_imgs, extrinsics, segs = render_images(sim, n)
    
    # create point cloud using open3d
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=depth_imgs[0].shape[1],
        height=depth_imgs[0].shape[0],
        fx=sim.camera.intrinsic.fx,
        fy=sim.camera.intrinsic.fy,
        cx=sim.camera.intrinsic.cx,
        cy=sim.camera.intrinsic.cy
    )
    
    scene = o3d.geometry.PointCloud()
    for depth, ex in zip(depth_imgs, extrinsics):
        depth_o3d = o3d.geometry.Image(depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, intrinsics, extrinsic = Transform.from_list(ex).as_matrix(), depth_scale=1.0, depth_trunc=2.0
        )
        scene += pcd
        
    scene = scene.voxel_down_sample(voxel_size=0.002)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([scene, coord])
    
    evaluate_grasp_point(sim, np.array([0.1, 0.1, 0.1]), np.array([0.0, 0.0, 1.0]))

    try:
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
