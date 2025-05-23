import os
import numpy as np
import open3d as o3d
import h5py
from data import read_ply, write_ply
from ssim import *
from util import *

ori_path='/home/lrx/pcc/mpeg-pcc-tmc2/data/8iVFBv2/soldier/Ply/'

rec1_base_path='/home/lrx/pcc/mpeg-pcc-tmc13/rec/soldier/predlift/r01/ply/'
rec2_base_path='/home/lrx/pcc/mpeg-pcc-tmc13/rec/soldier/predlift/r02/ply/'
rec3_base_path='/home/lrx/pcc/mpeg-pcc-tmc13/rec/soldier/predlift/r03/ply/'
rec4_base_path='/home/lrx/pcc/mpeg-pcc-tmc13/rec/soldier/predlift/r04/ply/'
rec5_base_path='/home/lrx/pcc/mpeg-pcc-tmc13/rec/soldier/predlift/r05/ply/'
rec6_base_path='/home/lrx/pcc/mpeg-pcc-tmc13/rec/soldier/predlift/r06/ply/'

h5_train='/home/lrx/pcd/GQE-Net/data_LT/h5/2048_2.0_predlift_seq3_cubeSearch/'
txt_path='/home/lrx/pcd/GQE-Net/matlab_patch_create/trainFile4.txt'
search_file = '/home/lrx/pcd/GQE-Net/matlab_patch_create/structural_similarity/ply/2048_2.0_predlift_seq3_cubeSearch'

rates = ['_r01','_r02', '_r03', '_r04', '_r05', '_r06']
patch_pts = 2048
seq_len = 3
overlap = 2.0
# 基准帧BASE
frame_id = 1

# 去除换行符
with open(txt_path, "r") as f:
    file = f.readlines()
    total_lines = len(file)
    for i in range(total_lines):
        file[i] = file[i].strip('\n')

for i in range(total_lines - seq_len + 1):
    # 读取连续seq_len帧的文件名
    ori_names = [file[i], file[i+1], file[i+2]]
    print('Process the %d-th sequence- %s \n' % (i, file[frame_id]))
    # 删除文件扩展名
    ori_onlyNames = [ori_names[0].split(".ply")[0], ori_names[1].split(".ply")[0], ori_names[2].split(".ply")[0]]
    print('ori_onlyNames', ori_onlyNames)
    write_name1 = ''.join([h5_train, ori_onlyNames[frame_id], rates[0], '.h5'])
    write_name2 = ''.join([h5_train, ori_onlyNames[frame_id], rates[1], '.h5'])
    write_name3 = ''.join([h5_train, ori_onlyNames[frame_id], rates[2], '.h5'])
    write_name4 = ''.join([h5_train, ori_onlyNames[frame_id], rates[3], '.h5'])
    write_name5 = ''.join([h5_train, ori_onlyNames[frame_id], rates[4], '.h5'])
    write_name6 = ''.join([h5_train, ori_onlyNames[frame_id], rates[5], '.h5'])
    
    # 添加比特率后缀
    rec_onlyNames1 = [x + rates[0] for x in ori_onlyNames]
    rec_onlyNames2 = [x + rates[1] for x in ori_onlyNames]
    rec_onlyNames3 = [x + rates[2] for x in ori_onlyNames]
    rec_onlyNames4 = [x + rates[3] for x in ori_onlyNames]
    rec_onlyNames5 = [x + rates[4] for x in ori_onlyNames]
    rec_onlyNames6 = [x + rates[5] for x in ori_onlyNames]

    # original
    ori_locs = [None] * seq_len
    ori_color_yuvs = [None] * seq_len
    # reconstructed
    rec_locs = [None] * seq_len
    rec_color_yuvs1 = [None] * seq_len
    rec_color_yuvs2 = [None] * seq_len
    rec_color_yuvs3 = [None] * seq_len
    rec_color_yuvs4 = [None] * seq_len
    rec_color_yuvs5 = [None] * seq_len
    rec_color_yuvs6 = [None] * seq_len
    

    for j in range(seq_len):
        ori = read_ply(ori_path + ori_names[j])
        # 存储原始点云连续seq_len帧的xyz
        ori_locs[j] = ori[:, :3].astype(np.float64)
        # 存储原始点云连续seq_len帧的yuv
        ori_color_yuvs[j] = rgb2yuv(ori[:, 3:]).astype(np.float64)
        rec1 = read_ply(rec1_base_path + rec_onlyNames1[j] + '.ply')
        rec2 = read_ply(rec2_base_path + rec_onlyNames2[j] + '.ply')
        rec3 = read_ply(rec3_base_path + rec_onlyNames3[j] + '.ply')
        rec4 = read_ply(rec4_base_path + rec_onlyNames4[j] + '.ply')
        rec5 = read_ply(rec5_base_path + rec_onlyNames5[j] + '.ply')
        rec6 = read_ply(rec6_base_path + rec_onlyNames6[j] + '.ply')
        # 存储重建点云连续seq_len帧的xyz
        rec_locs[j] = rec1[:, :3]
        # 存储重建点云连续seq_len帧的yuv
        rec_color_yuvs1[j] = rgb2yuv(rec1[:, 3:]).astype(np.float64)
        rec_color_yuvs2[j] = rgb2yuv(rec2[:, 3:]).astype(np.float64)
        rec_color_yuvs3[j] = rgb2yuv(rec3[:, 3:]).astype(np.float64)
        rec_color_yuvs4[j] = rgb2yuv(rec4[:, 3:]).astype(np.float64)
        rec_color_yuvs5[j] = rgb2yuv(rec5[:, 3:]).astype(np.float64)
        rec_color_yuvs6[j] = rgb2yuv(rec6[:, 3:]).astype(np.float64)
    
    # 以中间帧为代表(后续搜索对应点，创建patch)
    pointNumber = len(rec_locs[frame_id])
    patch_num = round(pointNumber * overlap / patch_pts)


    # 创建HDF5文件和数据集
    with h5py.File(write_name1, 'w') as f:
        f.create_dataset('/data', (patch_num, seq_len, patch_pts, 6))
        f.create_dataset('/label', (patch_num, seq_len, patch_pts, 3))
    with h5py.File(write_name2, 'w') as f:
        f.create_dataset('/data', (patch_num, seq_len, patch_pts, 6))
        f.create_dataset('/label', (patch_num, seq_len, patch_pts, 3)) 
    with h5py.File(write_name3, 'w') as f:
        f.create_dataset('/data', (patch_num, seq_len, patch_pts, 6))
        f.create_dataset('/label', (patch_num, seq_len, patch_pts, 3))
    with h5py.File(write_name4, 'w') as f:
        f.create_dataset('/data', (patch_num, seq_len, patch_pts, 6))
        f.create_dataset('/label', (patch_num, seq_len, patch_pts, 3))    
    with h5py.File(write_name5, 'w') as f:
        f.create_dataset('/data', (patch_num, seq_len, patch_pts, 6))
        f.create_dataset('/label', (patch_num, seq_len, patch_pts, 3))    
    with h5py.File(write_name6, 'w') as f:
        f.create_dataset('/data', (patch_num, seq_len, patch_pts, 6))
        f.create_dataset('/label', (patch_num, seq_len, patch_pts, 3))

    # 创建零数组
    box_label_train = np.zeros((patch_num, seq_len, patch_pts, 3))
    box_data_train1 = np.zeros((patch_num, seq_len, patch_pts, 6))
    box_data_train2 = np.zeros((patch_num, seq_len, patch_pts, 6))
    box_data_train3 = np.zeros((patch_num, seq_len, patch_pts, 6))
    box_data_train4 = np.zeros((patch_num, seq_len, patch_pts, 6))
    box_data_train5 = np.zeros((patch_num, seq_len, patch_pts, 6))
    box_data_train6 = np.zeros((patch_num, seq_len, patch_pts, 6))

    # 以基准帧为代表，使用FPS采样
    print('start farthest point sampling...')
    pt_locs = np.expand_dims(rec_locs[frame_id], 0)
    centroids_rec = farthest_point_sample(pt_locs, patch_num)
    # min_points = min(frame.shape[0] for frame in rec_locs)
    # out_of_bounds = centroids_rec > min_points
    # centroids_rec[out_of_bounds] = np.random.randint(0, min_points, centroids_rec.shape)[out_of_bounds]
    print('farthest point sampling finished...')
    
    
    # 基准帧的所有中心点坐标
    centroid_loc_base = rec_locs[frame_id][centroids_rec, :] # (1, 1734, 3)
    centroid_loc_base = np.squeeze(centroid_loc_base)
    # 初始化索引和距离的二维数组
    idxnn_rec_base = np.zeros((patch_num, patch_pts), dtype=int)
    rec_loc_base_pcd = o3d.geometry.PointCloud()
    rec_loc_base_pcd.points = o3d.utility.Vector3dVector(rec_locs[frame_id])
    kdtree_rec_base = o3d.geometry.KDTreeFlann(rec_loc_base_pcd)
    print('start search knn...')
    # 以FPS采样的中心点为中心，搜索patch_pts个最近邻点，形成对应的patch
    for a in range(patch_num):
        _, idxnn_rec_base[a], _ = kdtree_rec_base.search_knn_vector_3d(centroid_loc_base[a], patch_pts)
    print('search knn finished...')
    

    for m in range(patch_num):
        print('第%d/%d个patch...' % (m, patch_num))
        # 基准帧patch的几何和颜色
        curPatchIdx_rec_base = idxnn_rec_base[m, :]
        curPatchLoc_rec_base = rec_locs[frame_id][curPatchIdx_rec_base,:]
        curPatchCol_rec1_base = rec_color_yuvs1[frame_id][curPatchIdx_rec_base,:]
        curPatchCol_rec2_base = rec_color_yuvs2[frame_id][curPatchIdx_rec_base,:]
        curPatchCol_rec3_base = rec_color_yuvs3[frame_id][curPatchIdx_rec_base,:]
        curPatchCol_rec4_base = rec_color_yuvs4[frame_id][curPatchIdx_rec_base,:]
        curPatchCol_rec5_base = rec_color_yuvs5[frame_id][curPatchIdx_rec_base,:]
        curPatchCol_rec6_base = rec_color_yuvs6[frame_id][curPatchIdx_rec_base,:]

        # 存储基准帧的patch
        box_data_train1[m, frame_id, :, :] = np.hstack([curPatchLoc_rec_base, curPatchCol_rec1_base])
        box_data_train2[m, frame_id, :, :] = np.hstack([curPatchLoc_rec_base, curPatchCol_rec2_base])
        box_data_train3[m, frame_id, :, :] = np.hstack([curPatchLoc_rec_base, curPatchCol_rec3_base])
        box_data_train4[m, frame_id, :, :] = np.hstack([curPatchLoc_rec_base, curPatchCol_rec4_base])
        box_data_train5[m, frame_id, :, :] = np.hstack([curPatchLoc_rec_base, curPatchCol_rec5_base])
        box_data_train6[m, frame_id, :, :] = np.hstack([curPatchLoc_rec_base, curPatchCol_rec6_base])

        # corresponding idx in rec point cloud in the same patch with the original pt
        # 找到基准帧在原始点云中的索引, 是一一对应的
        idxnn_ori_base = np.zeros((patch_pts, 1), dtype=int)
        ori_loc_base_pcd = o3d.geometry.PointCloud()
        ori_loc_base_pcd.points = o3d.utility.Vector3dVector(ori_locs[frame_id])
        kdtree_ori_base = o3d.geometry.KDTreeFlann(ori_loc_base_pcd)
        for d in range(patch_pts):
            _, idxnn_ori_base[d], _ = kdtree_ori_base.search_knn_vector_3d(curPatchLoc_rec_base[d], 1)
        # curPatchLoc_ori_base = ori_locs[frame_id][idxnn_ori_base,:]
        curPatchCol_ori_base = ori_color_yuvs[frame_id][idxnn_ori_base,:]
        curPatchCol_ori_base = np.squeeze(curPatchCol_ori_base)
        # print('curPatchCol_ori_base: ', curPatchCol_ori_base.shape) # (2048, 3)
        box_label_train[m, frame_id, :, :] = curPatchCol_ori_base
        
        file_path = os.path.join(search_file, ori_onlyNames[frame_id])
        sub_dir_name = os.path.join(file_path, f"patchID{m}")
        pt0_name = os.path.join(sub_dir_name, f"patch{m}_frame_enhance.ply")
        os.makedirs(os.path.dirname(pt0_name), exist_ok=True)
        # 创建点云
        pt0 = np.hstack([curPatchLoc_rec_base, np.clip(np.round(yuv2rgb(curPatchCol_rec1_base)), 0, 255)]) # (2048, 6)
        # 保存点云
        write_ply(pt0, pt0_name)


        for n in range(seq_len):
            if n == frame_id:
                continue
            idxnn_rec_others = np.zeros((patch_num, patch_pts), dtype=int)
            # idxnn_rec_others = np.zeros((patch_pts, 1), dtype=int)
            rec_loc_others_pcd = o3d.geometry.PointCloud()
            rec_loc_others_pcd.points = o3d.utility.Vector3dVector(rec_locs[n])
            kdtree_rec_others = o3d.geometry.KDTreeFlann(rec_loc_others_pcd)
            idxnn_rec_others[m] = cube_search(kdtree_rec_base, kdtree_rec_others, curPatchLoc_rec_base, curPatchCol_rec1_base, rec_locs, 
                                              rec_color_yuvs1, ori_color_yuvs, centroid_loc_base[m], patch_pts, sub_dir_name, m, n, step_size=1.6)
            

            otherPatchIdx_rec = idxnn_rec_others[m, :]
            otherPatchLoc_rec = rec_locs[n][otherPatchIdx_rec,:]
            otherPatchCol_rec1 = rec_color_yuvs1[n][otherPatchIdx_rec,:]
            otherPatchCol_rec2 = rec_color_yuvs2[n][otherPatchIdx_rec,:]
            otherPatchCol_rec3 = rec_color_yuvs3[n][otherPatchIdx_rec,:]
            otherPatchCol_rec4 = rec_color_yuvs4[n][otherPatchIdx_rec,:]
            otherPatchCol_rec5 = rec_color_yuvs5[n][otherPatchIdx_rec,:]
            otherPatchCol_rec6 = rec_color_yuvs6[n][otherPatchIdx_rec,:]


            # 存储其他帧的patch
            box_data_train1[m, n, :, :] = np.hstack([otherPatchLoc_rec, otherPatchCol_rec1])
            box_data_train2[m, n, :, :] = np.hstack([otherPatchLoc_rec, otherPatchCol_rec2])
            box_data_train3[m, n, :, :] = np.hstack([otherPatchLoc_rec, otherPatchCol_rec3])
            box_data_train4[m, n, :, :] = np.hstack([otherPatchLoc_rec, otherPatchCol_rec4])
            box_data_train5[m, n, :, :] = np.hstack([otherPatchLoc_rec, otherPatchCol_rec5])
            box_data_train6[m, n, :, :] = np.hstack([otherPatchLoc_rec, otherPatchCol_rec6])
        
        
            # corresponding idx in rec point cloud in the same patch with the original pt
            # 找到其他帧在原始点云中的索引, 是一一对应的
            idxnn_ori_others = np.zeros((patch_pts, 1), dtype=int)
            ori_loc_others_pcd = o3d.geometry.PointCloud()
            ori_loc_others_pcd.points = o3d.utility.Vector3dVector(ori_locs[n])
            kdtree_ori_others = o3d.geometry.KDTreeFlann(ori_loc_others_pcd)
            for f in range(patch_pts):
                _, idxnn_ori_others[f], _ = kdtree_ori_others.search_knn_vector_3d(otherPatchLoc_rec[f], 1)
            otherPatchCol_ori = ori_color_yuvs[n][idxnn_ori_others,:]
            otherPatchCol_ori = np.squeeze(otherPatchCol_ori)
            box_label_train[m, n, :, :] = otherPatchCol_ori

    


    with h5py.File(write_name1, 'a') as f:
        f['/data'][:] = box_data_train1
        f['/label'][:] = box_label_train

    with h5py.File(write_name2, 'a') as f:
        f['/data'][:] = box_data_train2
        f['/label'][:] = box_label_train

    with h5py.File(write_name3, 'a') as f:
        f['/data'][:] = box_data_train3
        f['/label'][:] = box_label_train

    with h5py.File(write_name4, 'a') as f:
        f['/data'][:] = box_data_train4
        f['/label'][:] = box_label_train

    with h5py.File(write_name5, 'a') as f:
        f['/data'][:] = box_data_train5
        f['/label'][:] = box_label_train

    with h5py.File(write_name6, 'a') as f:
        f['/data'][:] = box_data_train6
        f['/label'][:] = box_label_train
