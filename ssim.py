import os
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors 
from dataclasses import dataclass
import numpy as np
import trimesh
import open3d as o3d
from data import write_ply
from ssim_score import ssim_score
from util import *
from typing import List, Tuple
import torch

base_frame = 2

def getKnnResult(data_A: trimesh.PointCloud, data_B: trimesh.PointCloud, k: int) -> Tuple[np.ndarray, np.ndarray]:
# def getKnnResult(data_A:"trimesh.PointCloud",data_B:"trimesh.PointCloud", k:int) -> "['np.ndarray', 'np.ndarray']":
    """
    Runs k-NN on datas and returns k-neighbors for each point with distances.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data_A.vertices)
    return neigh.kneighbors(data_B.vertices, return_distance=True)

@dataclass
class PointSSIM:
    ssimBA: float
    ssimAB: float
    ssimSym: float


def pc_ssim(pcA, pcB, params):
    # Formulation of neighborhoods in point clouds A and B
    distA, idA = getKnnResult(pcA, pcA, params.neighborhood_size)
    distB, idB = getKnnResult(pcB, pcB, params.neighborhood_size)
    # Association of neighborhoods between point clouds A and B
    _, idBA = getKnnResult(pcA, pcB, 1)
    _, idAB = getKnnResult(pcB, pcA, 1)

    out = {}

    if params.geom:
        geomQuantA = distA[:, 1:]
        geomQuantB = distB[:, 1:]
        
        # out["geomSSIM"] = PointSSIM(*ssim_score(geomQuantA, geomQuantB, idBA, idAB, params))
        out["geomSSIM"] = PointSSIM(*ssim_score(geomQuantA, geomQuantB, idA, idB, params))

    if params.color:
        # yA = pcA.colors[:, 0]
        # yB = pcB.colors[:, 0]
        yA = rgb2yuv(pcA.colors)[:, 0]
        yB = rgb2yuv(pcB.colors)[:, 0]
        colorQuantA = yA[idA]
        colorQuantB = yB[idB]
        
        out["colorSSIM"] = PointSSIM(*ssim_score(colorQuantA, colorQuantB, idA, idB, params))


    
    if params.normal:
        sumA = np.sum(np.multiply(idxWithArray(pcA.normal, idA), 
                                    np.tile(pcA.normal, (params.neighborhood_size, 1))), axis=1)
        secondPartA = np.multiply(np.sqrt(np.sum(idxWithArray(pcA.normal, idA)**2, axis=1)),
                                    np.sqrt(np.sum((np.tile(pcA.normal, (params.neighborhood_size, 1)))**2, axis=1)))
        nsA = np.real(1-(2*np.divide(np.arccos(np.abs(np.divide(sumA, secondPartA))),np.pi)))
        # Compansate nan values from 'arccos(pi/2)'
        nsA[np.isnan(nsA)] = 1 
        normQuantA = np.reshape(nsA, (-1,params.neighborhood_size), order='F')

        sumB = np.sum(np.multiply(idxWithArray(pcB.normal, idB), 
                                    np.tile(pcB.normal, (params.neighborhood_size, 1))), axis=1)
        secondPartB = np.multiply(np.sqrt(np.sum(idxWithArray(pcB.normal, idB)**2, axis=1)),
                                    np.sqrt(np.sum((np.tile(pcB.normal, (params.neighborhood_size, 1)))**2, axis=1)))
        nsB = np.real(1-(2*np.divide(np.arccos(np.abs(np.divide(sumB, secondPartB))),np.pi)))
        # Compansate nan values from 'arccos(pi/2)'
        nsB[np.isnan(nsB)] = 1 
        normQuantB = np.reshape(nsB, (-1,params.neighborhood_size), order='F')

        out["normSSIM"] = PointSSIM(*ssim_score(geomQuantA, geomQuantB, idBA, idAB, params))

    return out



def compute_ssim(args):
    # 多进程参数
    max_score, ii, centroid_loc_new, p_num, pt1, rec_loc, rec_color, params = args
    if max_score.value < 0.7096:
        rec_loc_pcd = o3d.geometry.PointCloud()
        rec_loc_pcd.points = o3d.utility.Vector3dVector(rec_loc)
        kdtree_rec = o3d.geometry.KDTreeFlann(rec_loc_pcd)
        
        # 多线程参数
        # ii, centroid_loc_new, p_num, pt1, kdtree_rec, rec_loc, rec_color, params = args
        
        # 初始化索引和距离的二维数组
        idxnn_rec_new = np.zeros((centroid_loc_new.shape[0], p_num), dtype=int)
        for c in range(centroid_loc_new.shape[0]):
            _, idxnn_rec_new[c], _ = kdtree_rec.search_knn_vector_3d(centroid_loc_new[c], p_num)
        
        curPatchIdx_rec = idxnn_rec_new[ii, :]
        curPatchLoc_rec = rec_loc[curPatchIdx_rec, :]
        curPatchCol_rec = rec_color[curPatchIdx_rec, :]
        pt2 = loadPointCloud(curPatchLoc_rec, curPatchCol_rec)
        pointSSIM = pc_ssim(pt1, pt2, params)
        score = 0.4*pointSSIM["geomSSIM"].ssimBA + 0.6*pointSSIM["colorSSIM"].ssimBA
        if score > max_score.value:
            max_score.value = score
        return score, curPatchLoc_rec, curPatchCol_rec
        # return score, curPatchIdx_rec
    else:
        return None
    


def cube_search(kdtree_rec_base, kdtree_rec_others, curPatchLoc_rec_base, curPatchCol_rec1_base, rec_loc, rec_color_yuvs, ori_color_yuvs, query_point, patch_pts, sub_dir_name, m, n, step_size):
    search_times = 3
    initial_step_size = step_size
    # 正方体搜索模式的偏移
    cube_offsets = [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]
    # cube_offsets = [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1),(0,0,0)]
    # 找到当前帧距离增强帧FPS采样点最近的点
    _, idx1, dis1 = kdtree_rec_others.search_knn_vector_3d(query_point, 1)
    curPatchLoc_rec1 = rec_loc[n][idx1]
    current_point = curPatchLoc_rec1[0]

    _, idx_tmp1, _ = kdtree_rec_base.search_knn_vector_3d(query_point, patch_pts)
    curPatchLoc_tmp1 = rec_loc[base_frame][idx_tmp1, :]
    curPatchCol_tmp1 = ori_color_yuvs[base_frame][idx_tmp1, :]
    
    while step_size >= initial_step_size / 2**(search_times-1):
        best_score = 0.0
        # best_score = float('inf')
        best_point = None
        best_score_idx = 0
        
        # 搜索当前点的周围27个方向
        for offset in cube_offsets:
            dx, dy, dz = offset
            # 计算d当前帧新的点的坐标
            new_point = [current_point[0] + dx * step_size, current_point[1] + dy * step_size, current_point[2] + dz * step_size]
            
            # _, idx_tmp1, _ = kdtree_rec_base.search_knn_vector_3d(query_point, 20)
            # curPatchCol_tmp1 = rec_color_yuvs[1][idx_tmp1, :]
            # _, idx_tmp2, _ = kdtree_rec_others.search_knn_vector_3d(new_point, 20)
            # curPatchCol_tmp2 = rec_color_yuvs[n][idx_tmp2, :]
            
            _, idx_tmp2, _ = kdtree_rec_others.search_knn_vector_3d(new_point, patch_pts)
            curPatchLoc_tmp2 = rec_loc[n][idx_tmp2, :]
            curPatchCol_tmp2 = ori_color_yuvs[n][idx_tmp2, :]
            # 搜索待查询立方体上每个点最近的patch_pts个点
            _, idx2, _ = kdtree_rec_others.search_knn_vector_3d(new_point, patch_pts)
            curPatchLoc_rec = rec_loc[n][idx2, :]
            curPatchCol_rec = rec_color_yuvs[n][idx2, :]
            # 以重建点云为基准和搜索
            patch_base = loadPointCloud(curPatchLoc_rec_base, curPatchCol_rec1_base)
            patch_searched = loadPointCloud(curPatchLoc_rec, curPatchCol_rec)
            # patch_base = np.hstack([curPatchLoc_rec_base, curPatchCol_rec1_base])
            # patch_searched = np.hstack([curPatchLoc_rec, curPatchCol_rec])
            # 以原始点云为基准和搜索
            ori_patch_base = loadPointCloud(curPatchLoc_tmp1, curPatchCol_tmp1)
            ori_patch_searched = loadPointCloud(curPatchLoc_tmp2, curPatchCol_tmp2)
            # normal1 = compute_normal(curPatchLoc_rec_base, query_point)
            # normal2 = compute_normal(curPatchLoc_rec, new_point)

            # 计算新的点与点云中的最近点的距离
            params = Params()
            pointSSIM = pc_ssim(ori_patch_base, ori_patch_searched, params)
            # score = 0.4*pointSSIM["geomSSIM"].ssimBA + 0.6*pointSSIM["colorSSIM"].ssimBA
            # score = pointSSIM["geomSSIM"].ssimBA
            score = pointSSIM["colorSSIM"].ssimBA
            # score = compute_chamfer_distance(curPatchLoc_rec_base, curPatchLoc_rec)
            # score = compute_angle_between_normals(normal1, normal2)
            # score = MSE(torch.tensor(curPatchCol_rec), torch.tensor(curPatchCol_rec1_base))
            # score = compute_average_difference(np.asarray(query_point), np.asarray(new_point))
            # score = np.linalg.norm(curPatchCol_tmp1 - curPatchCol_tmp2)

            # 保存未使用立方体搜索的patch
            if sub_dir_name != None and step_size == initial_step_size and dx == 0 and dy == 0 and dz == 0:
                pt_name = os.path.join(sub_dir_name, f"patch{m}_frame{n}_dis{dis1[0]}_score{score}_nosearch.ply")
                pt = np.hstack([curPatchLoc_rec, np.clip(np.round(yuv2rgb(curPatchCol_rec)), 0, 255)])
                # 保存点云
                write_ply(pt, pt_name)
            # 如果这个score更大，更新最大score和最佳点和idx
            if score > best_score:
                best_score = score
                best_point = new_point
                best_score_idx = idx2
                best_curPatchLoc_rec = curPatchLoc_rec
                best_curPatchCol_rec = curPatchCol_rec
            
        # 更新当前点为最佳点
        current_point = best_point
        # 减小步长
        step_size /= 2
    if sub_dir_name != None:
        # 保存最佳patch
        best_pt_name = os.path.join(sub_dir_name, f"patch{m}_frame{n}_bestScore{best_score}.ply")
        best_pt = np.hstack([best_curPatchLoc_rec, np.clip(np.round(yuv2rgb(best_curPatchCol_rec)), 0, 255)])
        write_ply(best_pt, best_pt_name)
    
    return best_score_idx


def full_search(kdtree_rec_base, kdtree_rec_others, rec_loc, rec_color_yuvs, ori_color_yuvs, query_point, patch_pts, sub_dir_name, m, n):
    # best_score = float('inf')
    best_score = 0.0
    # 找到当前帧距离增强帧FPS采样点最近的点
    _, idx1, dis1 = kdtree_rec_others.search_knn_vector_3d(query_point, 1)
    curPatchLoc_rec1 = rec_loc[n][idx1]
    current_point = curPatchLoc_rec1[0]
    _, idx_tmp1, _ = kdtree_rec_base.search_knn_vector_3d(query_point, patch_pts)
    curPatchLoc_rec_base = rec_loc[1][idx_tmp1, :]
    curPatchCol_rec_base = rec_color_yuvs[1][idx_tmp1, :]
    patch_base = loadPointCloud(curPatchLoc_rec_base, curPatchCol_rec_base)
    curPatchCol_tmp1 = ori_color_yuvs[1][idx_tmp1, :]
    _, idx_tmp2, dis_tmp2 = kdtree_rec_others.search_knn_vector_3d(current_point, patch_pts)
    curPatchLoc_nosearch = rec_loc[n][idx_tmp2, :]
    curPatchCol_nosearch = rec_color_yuvs[n][idx_tmp2, :]

    for i in range(round(0.05*patch_pts)):# 0.05+仅颜色效果不错
        _, idx, _ = kdtree_rec_others.search_knn_vector_3d(rec_loc[n][idx_tmp2[i], :], patch_pts)
        curPatchCol_tmp2 = ori_color_yuvs[n][idx, :]
        curPatchLoc_rec = rec_loc[n][idx, :]
        curPatchCol_rec = rec_color_yuvs[n][idx, :]
        patch_searched = loadPointCloud(curPatchLoc_rec, curPatchCol_rec)
        # score = np.linalg.norm(curPatchCol_tmp1 - curPatchCol_tmp2)
        params = Params()
        pointSSIM = pc_ssim(patch_base, patch_searched, params)
        # score = pointSSIM["geomSSIM"].ssimBA
        # score = pointSSIM["colorSSIM"].ssimBA
        # score = 0.2*pointSSIM["geomSSIM"].ssimBA + 0.8*pointSSIM["colorSSIM"].ssimBA
        score = compute_average_difference(curPatchCol_tmp1, curPatchCol_tmp2)
        if i == 0:
            pt_name = os.path.join(sub_dir_name, f"patch{m}_frame{n}_dis{dis_tmp2[0]}_score{score}_nosearch.ply")
            pt = np.hstack([curPatchLoc_nosearch, np.clip(np.round(yuv2rgb(curPatchCol_nosearch)), 0, 255)])
            # 保存点云
            write_ply(pt, pt_name)
        if score > best_score:
            best_score = score
            best_score_idx = idx
            best_curPatchCol_rec = curPatchCol_rec
            best_curPatchLoc_rec = curPatchLoc_rec

    best_pt_name = os.path.join(sub_dir_name, f"patch{m}_frame{n}_bestScore{best_score}.ply")
    best_pt = np.hstack([best_curPatchLoc_rec, np.clip(np.round(yuv2rgb(best_curPatchCol_rec)), 0, 255)])
    write_ply(best_pt, best_pt_name)
    
    return best_score_idx
