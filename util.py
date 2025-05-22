import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import random
import math
import numba
from scipy.spatial import distance
from sklearn.decomposition import PCA
import trimesh
from trimesh.caching import cache_decorator


def compute_average_difference(curPatchCol1, curPatchCol2):
    if curPatchCol1.shape[0] > 1 and curPatchCol2.shape[0] > 1:
        # 计算每个通道的平均值
        avg1 = np.mean(curPatchCol1, axis=0)
        avg2 = np.mean(curPatchCol2, axis=0)
        # 计算平方差值
        diff_square = np.square(avg1 - avg2)
        mean_diff = np.mean(diff_square)
    elif curPatchCol1.shape[0] == 1 and curPatchCol2.shape[0] == 1:
        diff_square = np.square(curPatchCol1 - curPatchCol2)
        mean_diff = np.mean(diff_square)
    return mean_diff

def compute_sad(point_cloud1, point_cloud2):
    # 确保两个点云的形状相同
    assert point_cloud1.shape == point_cloud2.shape, "Point clouds must have the same shape"
    # 计算两个点云之间的差值
    diff = point_cloud1 - point_cloud2
    # 取绝对值
    abs_diff = np.abs(diff)
    # 求和
    sad = np.sum(abs_diff)
    return sad


def compute_chamfer_distance(point_cloud1, point_cloud2):
    # 计算点云1中的每个点到点云2中最近点的距离
    dist1 = np.mean([np.min(distance.cdist(point_cloud1[i:i+1], point_cloud2, 'euclidean')) for i in range(point_cloud1.shape[0])])
    # 计算点云2中的每个点到点云1中最近点的距离
    dist2 = np.mean([np.min(distance.cdist(point_cloud2[i:i+1], point_cloud1, 'euclidean')) for i in range(point_cloud2.shape[0])])
    # 返回两个距离的平均值
    return (dist1 + dist2) / 2


def compute_normal(point_cloud, query_point, k=10):
    # 找到离query_point最近的k个点
    distances = np.sum((point_cloud - query_point)**2, axis=1)
    nearest_indices = np.argpartition(distances, k)[:k]
    nearest_points = point_cloud[nearest_indices]

    # 使用PCA找到主要方向
    pca = PCA(n_components=3)
    pca.fit(nearest_points)

    # 法向量是主成分中最小的那个
    normal = pca.components_[-1]

    return normal


def compute_angle_between_normals(normal1, normal2):
    # 计算两个法向量的点积
    dot_product = np.dot(normal1, normal2)

    # 计算两个法向量的模
    norm1 = np.linalg.norm(normal1)
    norm2 = np.linalg.norm(normal2)

    # 计算两个法向量之间的角度
    # angle = np.arccos(dot_product / (norm1 * norm2))
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)  # 保证cos_angle在[-1, 1]范围内
    angle = np.arccos(cos_angle)

    return angle


@numba.njit(numba.float64[:,:](numba.float64[:,:],numba.intc), cache=True, parallel=True) # numba.boolean[:]
def numbaFarthestPointDownSample(vertices, num_point_sampled):
    """ Use Farthest Point Sampling [FPS] to get a down sampled pointcloud
	INPUT:
            vertices: numpy array, shape (n,3) or (n,2)
            num_point_sampled: int, the desired number of points after down sampling
        OUTPUT:
            downSampledVertices: down sampled points with the original data type
	""" 
    N = vertices.shape[0]
    D = vertices.shape[1]
    assert num_point_sampled <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.empty((D,),np.float64)
    for d in range(D):
        _G[d] = np.mean(vertices[:,d])

    dists = np.zeros((N,),np.float64)
    for i in numba.prange(N):
        for d in range(D):
            dists[i] += (vertices[i,d] - _G[d])**2
    farthest = np.argmax(dists) 
    distances = np.inf * np.ones((N,))
    flags = np.zeros((N,), np.bool_)
    for _ in range(num_point_sampled):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        for i in numba.prange(N):
            dist = 0.
            if not flags[i]:
                for d in range(D):
                    dist += (vertices[i,d] - p_farthest[d])**2
            distances[i] = min(distances[i], dist)
        farthest = np.argmax(distances)
    return vertices[flags]


def slove_RT_by_SVD(src, dst):
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)

    src = src - src_mean # n, 3
    dst = dst - dst_mean
    H = np.transpose(src) @ dst

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T 

 
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T & U.T

    t = -R @ src_mean.T + dst_mean.T  # 3, 1

    return R, t


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def MSE(pred, gold):
    ''' Calculate MSE loss '''
    gold = gold.contiguous()
    loss_fn = torch.nn.MSELoss()

    loss = loss_fn(pred, gold)
    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = torch.tensor(xyz)    # from numpy to tensor
    xyz = xyz.to(torch.float)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids = centroids.detach().numpy()      # from tensor to numpy
    return centroids


def search_knn(c, x, k):
    pairwise_distance = torch.sum(torch.pow((c - x), 2), dim = -1)
    idx = (-pairwise_distance).topk(k = k, dim = -1)[1]   # (batch_size, num_points, k)
    return idx


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mse2psnr(mse):
    print('mse:', mse)
    psnr = 10*math.log10(255*255/mse)
    return psnr


def rgb2yuv(rgb):
    # PointNum=rgb.shape[0]
    yuv = np.zeros(rgb.shape)
    yuv[:, 0] = 0.2126*rgb[:, 0]+0.7152*rgb[:, 1]+0.0722*rgb[:, 2]
    yuv[:, 1] = -0.1146*rgb[:, 0]-0.3854*rgb[:, 1]+0.5000*rgb[:, 2] + 128
    yuv[:, 2] = 0.5000*rgb[:, 0]-0.4542*rgb[:, 1]-0.0458*rgb[:, 2] + 128
    # for i in range(PointNum):
    #     yuv[i, 0]=0.2126*rgb[i,0]+0.7152*rgb[i,1]+0.0722*rgb[i,2];
    #     yuv[i, 1]=-0.1146*rgb[i,0]-0.3854*rgb[i,1]+0.5000*rgb[i,2]+128;
    #     yuv[i, 2]=0.5000*rgb[i,0]-0.4542*rgb[i,1]-0.0458*rgb[i,2]+128;
    yuv = yuv.astype(np.float32)
    return yuv


def yuv2rgb(yuv):
    # PointNum=yuv.shape[0]
    yuv[:, 1] = yuv[:, 1] - 128
    yuv[:, 2] = yuv[:, 2] - 128
    rgb = np.zeros(yuv.shape)
    rgb[:, 0] = yuv[:, 0] + 1.57480 * yuv[:, 2]
    rgb[:, 1] = yuv[:, 0] - 0.18733 * yuv[:, 1] - 0.46813 * yuv[:, 2]
    rgb[:, 2] = yuv[:, 0] + 1.85563 * yuv[:, 1]
    return rgb

def cal_psnr(input1, input2):
    # img1 = input1.astype(np.float64)
    # img2 = input2.astype(np.float64)
    img1 = input1.to(torch.float64)
    img2 = input2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr



def eval_new(opt, model, input):
    model.eval()
    preds = model(input)
    return preds


def log_string(log, out_str):
    log.write(out_str + '\n')
    log.flush()
    print(out_str)



def cal_mean(list):       # 对于重复使用的点计算加权平均值
    number = len(list)
    idx = [index for index in range(number) if list[index].size != 1]       #  找出重复使用的点的索引
    for i in idx:
        i_temp = list[i]
        list[i] = torch.mean(i_temp, dim=0)
    return list


def repair_pc(ref_pc, repair_pc):
    # 创建一个KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ref_pc[:, :3])
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 对repair_pc中的每个点，找到在ref_pc中的最近点
    corresponding_points = []
    for point in repair_pc[:, :3]:
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        corresponding_point = ref_pc[idx[0]]
        corresponding_points.append(corresponding_point)

    repaired_pc = np.asarray(corresponding_points)
    return repaired_pc


# ------------------------------------- PointSSIM -------------------------------------
class BasePointCloud(trimesh.points.PointCloud):
    """
    We could have import normals with metadata however
    it is more convenient to have our base data class.
    """

    def __init__(self, mesh:trimesh.Trimesh, colors=None, metadata=None, **kwargs):
        vertices = mesh.vertices
        colors = mesh.visual.vertex_colors
        super().__init__(vertices, colors, metadata, **kwargs)
        self.mesh = mesh
        # Find normals at init to get rid of mesh.
        # self.normal

    # @cache_decorator
    # def normal(self):
    #     normals = self.mesh.vertex_normals
    #     self.__delattr__('mesh')
    #     return normals

def idxWithArray(a:np.ndarray, b:np.ndarray):
    """
    This is a feature in Matlab, I couldn't find any similar on Python
    So I wrote it myself for Numpy 2D arrays.
    
    Function tiles array 'a' in a loop of range(b.shape[-1])
    and sorts a with b[:,idx] before concatenation.

    INPUTS
        a: np.ndarray, shape of [n x m]: vertex_normals of point cloud for this matter.
        b: np.ndarray, shape of [n x p] k-neighbors indexes for every point for this matter. 
            p is the number of neighbors for every point.
    OUTPUT
        out: np.ndarray, shape of [n*p x m]
    """
    out = np.zeros((0,3), dtype=np.float64)
    if a.shape[0] == b.shape[0]:
        for idx in range(b.shape[-1]):
            arr = a[b[:,idx]]
            out = np.concatenate((out,arr), axis=0)
    return out


   
def loadPointCloud(points, colors, **kwargs):
    # data = trimesh.load(filepath)
    data = trimesh.PointCloud(vertices=points, colors=colors)
    pointCloud = BasePointCloud(mesh=data, **kwargs)
    return pointCloud


class Params:
    def __init__(self):
        self.geom = True
        self.normal = False
        self.curvature = False
        self.color = True
        self.estimator = "MeanAD"
        self.pooling_type = "Mean"
        self.neighborhood_size = 12
        self.constant = 2.2204e-16
        self.ref = 1


def pooling(qMap, poolingType):
    if poolingType == 'Mean':
        score = np.nanmean(qMap)
    elif poolingType == 'MSE':
        score = np.nanmean(qMap**2)
    elif poolingType == 'RMS':
        score = np.sqrt(np.nanmean(qMap**2))
    else:
        raise Exception('Wrong pooling type...')
    return score


def feature_map(quant:np.ndarray, estType:list):
    """
    In Matlab version you can choose more than one estimator type
    However due to simplicity reasons, you can choose only one here.
    
    INPUTS
        quant: Per-attribute quantities that reflect corresponding local
            properties of a point cloud. The size is LxK, with L the number
            of points of the point cloud, and K the number of points
            comprising the local neighborhood.
        estType: Defines the estimator(s) that will be used to
            compute statistical dispersion, with available options:
            {'STD', 'VAR', 'MeanAD', 'MedianAD', 'COV', 'QCD'}.
            More than one options can be enabled.

    OUTPUTS
        fMap: Feature map of a point cloud, per estimator. The size is LxE,
            with L the number of points of the point cloud and E the length
            of the 'estType'.
    """

    fMap = np.zeros((np.size(quant,0), 1), np.float64)

    if estType == "STD":
        fMap = np.std(quant, 1, ddof=1)
    elif estType == "VAR":
        fMap = np.var(quant, 1, ddof=1)
    elif estType == "MeanAD":
        fMap = np.mean(abs(quant - np.mean(quant, 1).reshape(-1,1)), 1)
    elif estType == "MedianAD":
        fMap = np.median(abs(quant - np.median(quant, 1).reshape(-1, 1)), 1)
    elif estType == "COV":
        fMap = np.std(quant, 1, ddof=1) / np.mean(quant, 1)
    elif estType == "QCD":
        """
        There are no equivalent of Matlab's quantile on Python 3.7 
        
        However on latest Numpy versions there is an option called 'method',
        which is a replacement of 'interpolation' option that can take more 
        arguments than interpolation.
        """
        qq = np.quantile(quant, [.25, .75], 1, interpolation='linear')
        fMap = (qq[1,:] - qq[0,:]) / (qq[1,:] + qq[0,:])
    return fMap


def error_map(fMapY, fMapX, idYX, CONST):
    fMapX = np.reshape(fMapX, (-1,1))
    fMapY = np.reshape(fMapY, (-1,1))

    # print((fMapX[idYX]).shape)
    # print(fMapY.shape)
    # print(idYX.shape)
    # eMapYX = np.divide((np.abs(fMapX[idYX].reshape(-1,1) - fMapY)).reshape(-1,1), 
    #             (np.max(np.concatenate((np.abs(fMapX[idYX].reshape(-1,1)), np.abs(fMapY)), axis=1), axis=1) + CONST).reshape(-1,1))

    # print(np.concatenate((np.abs(a), np.abs(b)), axis=1).shape)
    # print(np.max(np.concatenate((np.abs(a), np.abs(b)), axis=1), axis=1).reshape(-1, 1).shape)
    
    a = np.squeeze(fMapX[idYX])
    b = fMapY
    c = np.max(np.concatenate((np.abs(a), np.abs(b)), axis=1), axis=1).reshape(-1, 1)
    
    eMapYX = np.divide((np.abs(a - b)), (c + CONST))
    return eMapYX


if __name__ == "__main__":
    # c = torch.randn(2,3)
    # x = torch.randn(5,3)
    # print(x, c)
    # idx = search_knn(c, x, 1)
    # print(idx.size())
    # print(x[idx])
    # print(torch.sum(x[idx]-c))
    # print(x[idx].size())

    # print(np.clip(np.round(yuv2rgb(c)), 0, 255))
    
    arr1 = np.array([[110, 86, 66],[137, 109, 79]])
    print(rgb2yuv(arr1))
    
    arr2 = np.array([[81.7196, 137.31241, 130.084],[105.760994, 122.739395, 133.8664]])
    print(np.clip(np.round(yuv2rgb(arr2)), 0, 255))

    # arr3 = np.array([[54.948833, 55.002335, 75.596924],[56.056713, 55.540752, 75.187256]])
    # print(yuv2rgb(arr3))