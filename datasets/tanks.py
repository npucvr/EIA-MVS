from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy


class MVSDataset(Dataset):
    def __init__(self, datapath, n_views=7, split='intermediate'):
        self.levels = 4
        self.datapath = datapath
        self.split = split
        self.build_metas()
        self.n_views = n_views

    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            # self.scans = ['Family', 'Horse','Playground', 'Francis',  'Train', 'Lighthouse', 'M60', 'Panther']
            # self.scans = ['Playground', 'Francis',  'Train', 'Lighthouse', 'M60', 'Panther']
            self.scans = ['Horse']

            
        elif self.split == 'advanced':
            self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
                          'Museum', 'Palace', 'Temple']

        elif self.split == 'train':
            self.scans = ['Caterpillar', 'Ignatius', 'Meetingroom', 'Truck']

        for scan in self.scans:
            pair_path = os.path.join(self.datapath, self.split, scan, 'pair_new.txt')
            if not os.path.exists(pair_path):
                pair_path = os.path.join(self.datapath, self.split, scan, 'pair.txt')
            if self.split == 'train':
                pair_path = os.path.join(self.datapath, "TAT_training_set_input", scan, 'pair.txt')
            with open(pair_path) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        self.metas += [(scan, -1, ref_view, src_views)]
   

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])
        if len(lines[11].split()) >= 3:
            depth_max = float(lines[11].split()[3])


        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def scale_input(self, intrinsics, img):
        """
        intrinsics: 3x3
        img: W H C
        """
        intrinsics[1,2] =  intrinsics[1,2] - 28  # 1080 -> 1024
        img = img[28:1080-28, :, :]
        return intrinsics, img

    def __len__(self):
        return len(self.metas)
    

        
    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base
        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics


    def __getitem__(self, idx):
        scan, _, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]
        imgs = []

        # src_views_near_low = min(max(ref_view - (self.n_views-1)//2, 0), 140)
        # # src_views_near_high = src_views_near_low + self.n_views
        # src_views_new = []
        # for i in range(self.n_views):
        #     if (src_views_near_low + i) != ref_view:
        #         src_views_new.append(src_views_near_low + i)
        #
        # view_ids = [ref_view] + src_views_new

        # depth = None
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, self.split, scan, f'images/{vid:08d}.jpg')
            if self.split == 'train':
                img_filename = os.path.join(self.datapath, "TAT_training_set_input", scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.datapath, self.split, scan, f'cams_1/{vid:08d}_cam.txt')
            if self.split == 'train':
                proj_mat_filename = os.path.join(self.datapath, "TAT_training_set_input", scan, f'cams/{vid:08d}_cam.txt')

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            # if scan == 'Panther' or scan == 'Playground':
            if scan == 'Panther':
                img, intrinsics = self.scale_mvs_input(img, intrinsics, 1216, 896)
            else:
                intrinsics, img = self.scale_input(intrinsics, img)
            # if scan == 'Horse':
            #     depth_min_ = depth_min_ * 2 / 3
            #     depth_max_ = depth_max_ * 2 / 3
            imgs.append(img.transpose(2,0,1))

            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_3 = np.zeros(shape=(2, 4, 4), dtype=np.float32)

            intrinsics[:2,:] *= 0.125
            proj_mat_0[0,:4,:4] = extrinsics.copy()
            proj_mat_0[1,:3,:3] = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_1[0,:4,:4] = extrinsics.copy()
            proj_mat_1[1,:3,:3] = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_2[0,:4,:4] = extrinsics.copy()
            proj_mat_2[1,:3,:3] = intrinsics.copy()

            intrinsics[:2,:] *= 2
            proj_mat_3[0,:4,:4] = extrinsics.copy()
            proj_mat_3[1,:3,:3] = intrinsics.copy()  

            proj_matrices_0.append(proj_mat_0)
            proj_matrices_1.append(proj_mat_1)
            proj_matrices_2.append(proj_mat_2)
            proj_matrices_3.append(proj_mat_3)

            if i == 0:  # reference view
                depth_min =  depth_min_
                depth_max = depth_max_


        # proj_matrices: N*4*4
        proj={}
        proj['stage1'] = np.stack(proj_matrices_0)
        proj['stage2'] = np.stack(proj_matrices_1)
        proj['stage3'] = np.stack(proj_matrices_2)
        proj['stage4'] = np.stack(proj_matrices_3)




        return {"imgs": imgs, # N*3*H0*W0
                "proj_matrices": proj, # N*4*4
                "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
