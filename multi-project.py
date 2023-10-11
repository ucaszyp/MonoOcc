import os
from PIL import Image
import yaml
import numpy as np
from numpy.linalg import inv

class KITTIDATASET():
    def __init__(
        self,
        split,
        data_root,
        temporal = [],
    ):
        self.data_root = data_root
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split 
        self.sequences = splits[split]
        with open(os.path.join(self.data_root, "dataset", "semantic-kitti.yaml"), 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.learning_map = semkittiyaml['learning_map']

        self.n_classes = 20
        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            # proj_matrix_2 = P2 @ T_velo_2_cam
                
            image_dir = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2"
            )
            image_list = sorted(os.listdir(image_dir))
            for image_file in image_list:
            
                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P2": P2,
                        "T_velo_2_cam": T_velo_2_cam,
                        "image_file": image_file
                    }
                )

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        example = self.get_data_info(index)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]
        sequence = scan["sequence"]
        image_path = scan["image_file"]
        filename = os.path.basename(image_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id, image_path)
        img_list, sem_list, lidar_list = self.get_input_info(sequence, frame_id)        
        data_info = dict(
            img_metas = meta_dict,
            img = img_list,
            sem = sem_list,
            lidar = lidar_list
        )
        return data_info

    def get_meta_info(self, scan, sequence, frame_id, image_path):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,
            image_path (str): image path.

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []

        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P2 = scan["P2"]
        cam_k = P2[0:3, 0:3]
        intrinsic = cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = np.identity(4)
        lidar2img_rt[:3, :] = (P2 @ lidar2cam_rt)
        # lidar2img_rt = (viewpad @ lidar2cam_rt)

        lidar2img_rts.append(lidar2img_rt)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_intrinsics.append(intrinsic)
        image_paths.append(rgb_path)

        # for reference img
        seq_len = len(self.poses[sequence])
        
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )

            pose_list = self.poses[sequence]

            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            ref2target = np.matmul(inv(target), ref) # both for lidar

            target2cam = scan["T_velo_2_cam"] # lidar to camera
            ref2cam = target2cam @ ref2target

            lidar2cam_rt  = ref2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)

            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt)
            cam_intrinsics.append(intrinsic)
            image_paths.append(rgb_path)

        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list = []
        sem_list = []
        lidar_list = []
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        sem_path = rgb_path.replace("image_2", "sem_gt")
        
        # add lidar
        lidar_path = rgb_path.replace("image_2", "velodyne").replace(".png", ".bin")
        label_path = rgb_path.replace("image_2", "labels").replace(".png", ".label")
        
        img = Image.open(rgb_path).convert("RGB")
        sem = Image.open(sem_path)
        sem = np.array(sem)
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        image_list.append(img)
        sem_list.append(sem)
        
        lidar = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape((-1, 4))
        label = np.fromfile(label_path, dtype=np.uint32, count=-1)
        label = label & 0xFFFF
        label = np.vectorize(self.learning_map.__getitem__)(label)
        lidar[:,-1] = label
        lidar_list.append(lidar)
        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )
            lidar_path = rgb_path.replace("image_2", "velodyne").replace(".png", ".bin")
            label_path = rgb_path.replace("image_2", "labels").replace(".png", ".label")
            lidar = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape((-1, 4))
            label = np.fromfile(label_path, dtype=np.uint32, count=-1)
            label = label & 0xFFFF
            label = np.vectorize(self.learning_map.__getitem__)(label)
            lidar[:,-1] = label
            lidar_list.append(lidar)

            sem_path = rgb_path.replace("image_2", "sem_gt")
            img = Image.open(rgb_path).convert("RGB")
            sem = Image.open(sem_path)
            sem = np.array(sem)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            image_list.append(img)
            sem_list.append(sem)
        
        return image_list, sem_list, lidar_list

def cam2image(points, lidar2img):
    ndim = points.ndim
    points = points.T
    points_proj = np.matmul(lidar2img.reshape([4,4]), points)
    depth = points_proj[2,:]
    depth[depth == 0] = -1e-6
    u = np.round(points_proj[0, :] / np.abs(depth)).astype(np.int)
    v = np.round(points_proj[1, :] / np.abs(depth)).astype(np.int)

    if ndim == 2:
        u = u[0]
        v = v[0]
        depth = depth[0]
    return u, v, depth

if __name__ == "__main__":
    split = "train"
    dataroot = "./kitti/"
    temporal = [-4,-3,-2,-1]
    kitti_ds = KITTIDATASET(split, dataroot, temporal)
    for i in range(20, 100):
        data_info = kitti_ds.prepare_data(i)
        meta_dict = data_info["img_metas"]
        file_names = meta_dict["img_filename"]
        frame_id = meta_dict["frame_id"]
        lidar_list = data_info["lidar"]
        img_list = data_info["img"]
        lidar2img_list = meta_dict["lidar2img"]
        for i in range(len(lidar_list)):
            points = lidar_list[i]
            lidar2img = lidar2img_list[i]
            u, v, depth = cam2image(points, lidar2img)


