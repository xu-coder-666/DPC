from __future__ import print_function
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Pool
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import *
from model_GQE_Net_final import GAPCN
from dpc_net import dpc
from tqdm import tqdm
import datetime
import trimesh
import time
import open3d as o3d
from torch.utils.data import DataLoader, TensorDataset
from util import *
from ssim import *
from sewar.full_ref import psnr
from tensorboardX import SummaryWriter

devices = "cuda:"
PT_NUM = 1024
SEQ_LEN = 3
enhance_frame = 2 # 增强帧，最后一帧-1或中间帧1
base_frame = 2 # 基准帧，用于cubesearch
overlap = 2.0
normalize = False
search = True


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('copy main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('copy model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('copy util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('copy data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    daytime = datetime.datetime.now().strftime('%Y-%m-%d')  # year,month,day
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    channel = args.train_channel
    yuv_list = ['y', 'u', 'v']
    DATA_DIR = os.path.join(BASE_DIR, args.train_h5_txt)
    DATA_DIR_TEST = os.path.join(BASE_DIR, args.valid_h5_txt)
    logs_path = args.log_path + '/GQE-Net/' + daytime + '/' + yuv_list[channel]
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model_path = args.pth_path + '/GQE-Net/' + daytime + '/' + yuv_list[channel]
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    txt_loss_mse = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_loss.txt'
    txt_mse_path = os.path.join(logs_path, txt_loss_mse)
    txt_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_loss_rgb.txt'
    txt_psnr_path = os.path.join(logs_path, txt_loss_psnr)
    txtValid_name_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_lossValid.txt'
    txt_lossValid_path = os.path.join(logs_path, txtValid_name_loss)
    txt_valid_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_valid_loss_rgb.txt'
    txt_valid_psnr_path = os.path.join(logs_path, txt_valid_loss_psnr)

    traindata, label = load_h5(DATA_DIR, normalize=normalize)
    dataset = TensorDataset(traindata, label)
    train_loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True)

    testdata, label = load_h5(DATA_DIR_TEST, normalize=normalize)
    dataset_test = TensorDataset(testdata, label)
    test_loader = DataLoader(dataset=dataset_test,
                            batch_size=args.test_batch_size,
                            shuffle=True,
                            drop_last=True)

    writer_date = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    train_writer_path = os.path.join('./tensorboard', writer_date, yuv_list[channel], 'train')
    train_writer = SummaryWriter(train_writer_path)
    test_writer_path = os.path.join('./tensorboard', writer_date, yuv_list[channel], 'test')
    test_writer = SummaryWriter(test_writer_path)

    device = torch.device(devices + args.gpu if args.cuda else "cpu")
    # model = GAPCN(devices + args.gpu).to(device)
    model = dpc(devices + args.gpu).to(device)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    if args.has_model:
        checkpoint = torch.load(args.model1_path)
        if channel == 1:
            checkpoint = torch.load(args.model2_path)
        elif channel == 2:
            checkpoint = torch.load(args.model3_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = MSE
    max_psnr = 0
    for epoch in range(args.epochs):
        epoch_loss = AverageMeter()
        epoch_loss_ori = AverageMeter()
        ####################
        # Train
        ####################
        args.lr = args.lr * (0.25 ** (epoch // 60)) # 60
        for p in opt.param_groups:
            p['lr'] = args.lr
        train_loss = 0.0
        count = 0.0
        model.train()
        with tqdm(total=(traindata.shape[0] - traindata.shape[0] % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.epochs))
            for data, label in train_loader:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]
                # qp = torch.zeros(batch_size)
                # for ii in range(batch_size):
                #     qp[ii] = data[ii, 0, 6]
                
                
                data = torch.cat((data[:, :, :, :3], torch.unsqueeze(data[:, :, :, channel + 3], dim=-1)), dim=-1) # xyz坐标 + Y/U/V
                label = label[:, :, :, channel] # 只取 Y/U/V 其中一个通道

                if len(label.size()) == 3:
                    label = torch.unsqueeze(label, dim=-1)
                data = data.permute(0, 1, 3, 2) # [2, 4, 4, 2048]

                opt.zero_grad()
                # ********************************************** 取增强帧的数据 **********************************************
                rec = data.permute(0, 1, 3, 2)[:, enhance_frame, :, 3:] # [2, 4, 2048, 1]    [2, 1024, 1]
                data = torch.autograd.Variable(data, requires_grad=True)
                logits = model(data)
                # *********************************************** 观察一下logits预测结果是否有太大偏差 *****************************
                # print('rec:', rec)
                # print('logits:', logits)
                # *********************************************** 观察一下logits预测结果是否太大偏差 *****************************
                
                # *********************************************** 取增强帧的标签 **********************************************
                label = label[:, enhance_frame, :, :] # [2, 4, pts]  [2, 1024, 1]
                # print('label:', label)
                loss = criterion(logits, label)
                loss_ori = criterion(rec, label)
                loss.backward()
                opt.step()
                epoch_loss.update(loss.item(), len(rec))
                epoch_loss_ori.update(loss_ori.item(), len(rec))

                count += batch_size
                train_loss += loss.item() * batch_size
                
                _tqdm.set_postfix(loss='{:.7f}'.format(epoch_loss.avg))
                _tqdm.update(len(data))

                train_writer.add_scalar('Loss/train', epoch_loss.avg, epoch)
            
        scheduler.step()
        
        epoch_psnr = mse2psnr(epoch_loss.avg)
        epoch_psnr_ori = mse2psnr(epoch_loss_ori.avg)
        file_log = open(txt_mse_path, 'a')
        print('epoch:{} loss:{} loss_ori:{}'.format(epoch, epoch_loss.avg, epoch_loss_ori.avg), file=file_log)
        file_log.close()
        print('epoch:{}'.format(epoch), 'average loss:{}'.format(epoch_loss.avg))
        file_loss = open(txt_psnr_path, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_psnr), 'psnr_origin:{}'.format(epoch_psnr_ori), file=file_loss)

        file_loss.close()

        ####################
        # Test / Validating...
        ####################
        print('epoch:%d   validating...' % epoch)
        valid_epoch_loss = AverageMeter()
        valid_epoch_loss_ori = AverageMeter()

        model.eval()
        with tqdm(total=(testdata.shape[0] - testdata.shape[0] % args.test_batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, args.epochs))
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]
                # qp = torch.zeros(batch_size)
                # for ii in range(batch_size):
                #     qp[ii] = data[ii, 0, 6]
                
                data = torch.cat((data[:, :, :, :3], torch.unsqueeze(data[:, :, :, channel + 3], dim=-1)), dim = -1)
                label = label[:, :, :, channel]
                
                if len(label.size()) == 3:
                    label = torch.unsqueeze(label, dim=-1)

                data = data.permute(0, 1, 3, 2)
                with torch.no_grad():
                    logits = model(data)
                # ********************************************** 取增强帧的数据 **********************************************
                rec = data.permute(0, 1, 3, 2)[:, enhance_frame, :, 3:]

                # ********************************************** 取增强帧的标签 **********************************************
                label = label[:, enhance_frame, :, :]

                # *********************************************** 观察一下logits预测结果是否有太大偏差 *****************************
                # print('label:', label)
                # print('val rec:', rec)
                # print('val logits:', logits)
                # print(label.size(), rec.size(), logits.size())
                # *********************************************** 观察一下logits预测结果是否太大偏差 *****************************

                loss = criterion(logits, label)
                loss_ori = criterion(rec, label)

                valid_epoch_loss.update(loss.item(), len(data))
                valid_epoch_loss_ori.update(loss_ori.item(), len(data))

                _tqdm.set_postfix(loss='{:.7f}'.format(valid_epoch_loss.avg))
                _tqdm.update(len(data))

        valid_epoch_psnr = mse2psnr(valid_epoch_loss.avg)
        valid_epoch_psnr_ori = mse2psnr(valid_epoch_loss_ori.avg)
        test_writer.add_scalar('PSNR/valid', valid_epoch_psnr, epoch)
        # io.cprint(outstr)
        print('valid loss_ori:{}'.format(valid_epoch_loss_ori.avg), 'valid loss_preds:{}'.format(valid_epoch_loss.avg))
        fileValid_loss = open(txt_lossValid_path, 'a')
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss.avg))
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss.avg), 'valid average loss_ori:{}'.\
              format(valid_epoch_loss_ori.avg), file=fileValid_loss)

        fileValid_loss.close()
        file_valid_psnr = open(txt_valid_psnr_path, 'a')
        print('epoch:{}'.format(epoch), 'valid_psnr:{}'.format(valid_epoch_psnr), 'valid_psnr_origin:{}'.format(valid_epoch_psnr_ori),
              file=file_valid_psnr)
        
        if epoch % 5 == 0 or valid_epoch_psnr-valid_epoch_psnr_ori > max_psnr:  # save the model with max PSNR promotion
            max_psnr = valid_epoch_psnr - valid_epoch_psnr_ori
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss,
            },
                '%s/model_%d.pth' % (model_path, epoch)
            )

        file_valid_psnr.close()



def test(args, io):
    # ****************************************路径配置 导入模型*******************************************************
    device = torch.device(devices + args.gpu)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR_TEST = os.path.join(BASE_DIR, args.test_ply_txt)
    ori_path = os.path.join(BASE_DIR, args.test_ori_ply)
    rec_path = os.path.join(BASE_DIR, args.test_rec_ply)

    # model1 = GAPCN(devices + args.gpu).to(device)
    model1 = dpc(devices + args.gpu).to(device)
    # checkpoint = torch.load(args.model1_path, map_location={'cuda:1': 'cuda:0'})
    checkpoint = torch.load(args.model1_path, map_location=torch.device(devices + args.gpu))
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1 = model1.eval()

    # model2 = GAPCN(devices + args.gpu).to(device)
    model2 = dpc(devices + args.gpu).to(device)
    # checkpoint = torch.load(args.model2_path, map_location={'cuda:1': 'cuda:0'})
    checkpoint = torch.load(args.model2_path, map_location=torch.device(devices + args.gpu))
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2 = model2.eval()

    # model3 = GAPCN(devices + args.gpu).to(device)
    model3 = dpc(devices + args.gpu).to(device)
    # checkpoint = torch.load(args.model3_path, map_location={'cuda:1': 'cuda:0'})
    checkpoint = torch.load(args.model3_path, map_location=torch.device(devices + args.gpu))
    model3.load_state_dict(checkpoint['model_state_dict'])
    model3 = model3.eval()

    textfile_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_test_ply_trained_GAPCN.txt'
    LOG_FOUT = open(os.path.join(args.log_path_test, textfile_name), 'w')
    LOG_FOUT.write(str(args) + '\n')
    # ****************************************路径配置 导入模型*******************************************************

    p_num = PT_NUM
    iter = 0
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)
    # 去除换行符
    with open(DATA_DIR_TEST, "r") as f:
        # for line in f.readlines():
        lines = f.readlines()
        total_lines = len(lines)
        for i in range(total_lines):
            lines[i] = lines[i].strip('\n')
        
    for i in range(total_lines):
        t11 = time.time()
        # lines[i] = lines[i].strip('\n')
        log_string(LOG_FOUT, 'enhance sequence: %s:' % (lines[i+enhance_frame]))
        iter = iter + 1
        log_string(LOG_FOUT, 'It uses sequence: %s ~ %s, iter: %d' % (lines[i], lines[i+SEQ_LEN-1], iter))
        if os.path.splitext(lines[i])[1] == ".ply":
            # ori_name = lines[i].split('_r0')[0] + ".ply"
            # 原始点云name
            ori_names = [line.split('_r0')[0] + ".ply" for line in lines[i : i+SEQ_LEN]]
            print('ori_names', len(ori_names))
            pointcloud_ori = [0] * SEQ_LEN
            ori_color      = [0] * SEQ_LEN
            pointcloud_rec = [0] * SEQ_LEN
            rec_loc        = [0] * SEQ_LEN
            rec_color      = [0] * SEQ_LEN


            for j in range(SEQ_LEN):
                pointcloud_ori[j] = read_ply(os.path.join(ori_path, ori_names[j]))  # [numpoint, 6]
                pointcloud_rec[j] = read_ply(os.path.join(rec_path, lines[j + i]))
                pointcloud_ori[j] = repair_pc(pointcloud_ori[j], pointcloud_rec[j])

                ori_color[j] = rgb2yuv(pointcloud_ori[j][:, 3:]).astype(np.float32)
                # ***************************** 使用rgb颜色空间 *******************************
                # ori_color[j] = np.array(pointcloud_ori[j][:, 3:]).astype(np.float32)
                # ***************************** 使用rgb颜色空间 *******************************
                rec_loc[j] = pointcloud_rec[j][:, :3]  # [numpoint, 3],location
                rec_color[j] = rgb2yuv(pointcloud_rec[j][:, 3:]).astype(np.float32) # [numpoint, 3],color
                # ***************************** 使用rgb颜色空间并归一化用于预测 *******************************
                # rec_color[j] = np.array(pointcloud_rec[j][:, 3:]).astype(np.float32) / 255
                # ***************************** 使用rgb颜色空间并归一化用于预测  *******************************

            # 以最后一帧为代表，计算patch个数
            pointNum = rec_loc[enhance_frame].shape[0]
            numPatch = int(pointNum * overlap // p_num)

            # process_time_sum = 0
            fps_time_begin = time.time()

            # 以base帧为代表，使用FPS采样
            if search:
                pt_locs = np.expand_dims(rec_loc[base_frame], 0)
            else:
                pt_locs = np.expand_dims(rec_loc[enhance_frame], 0)
            idx = farthest_point_sample(pt_locs, numPatch)
            # min_points = min(frame.shape[0] for frame in rec_loc)
            # out_of_bounds = idx > min_points
            # idx[out_of_bounds] = np.random.randint(0, min_points, idx.shape)[out_of_bounds]

            # 基准帧的所有中心点坐标
            if search:
                centroid_loc_base = rec_loc[base_frame][idx, :] # (1, 1734, 3)
            else:
                centroid_loc_base = rec_loc[enhance_frame][idx, :] 
            centroid_loc_base = np.squeeze(centroid_loc_base)
            rec_loc_base_pcd = o3d.geometry.PointCloud()
            if search:
                rec_loc_base_pcd.points = o3d.utility.Vector3dVector(rec_loc[base_frame])
            else:
                rec_loc_base_pcd.points = o3d.utility.Vector3dVector(rec_loc[enhance_frame])
            
            kdtree_rec_base = o3d.geometry.KDTreeFlann(rec_loc_base_pcd)
            rec_loc_pcd0 = o3d.geometry.PointCloud()
            rec_loc_pcd0.points = o3d.utility.Vector3dVector(rec_loc[0])
            kdtree_rec0 = o3d.geometry.KDTreeFlann(rec_loc_pcd0)
            rec_loc_pcd1 = o3d.geometry.PointCloud()
            rec_loc_pcd1.points = o3d.utility.Vector3dVector(rec_loc[1])
            kdtree_rec1 = o3d.geometry.KDTreeFlann(rec_loc_pcd1)
            kdtree_rec = [kdtree_rec0, kdtree_rec1, kdtree_rec_base]

            # 初始化索引和距离的二维数组
            idxnn_rec_base = np.zeros((centroid_loc_base.shape[0], p_num), dtype=int)
            # dis = np.zeros((centroid_loc_base.shape[0], p_num), dtype=float)
            print('k nearest neighbor doing...')
            # test_time_start = time.time()
            for a in range(centroid_loc_base.shape[0]):
                _, idxnn_rec_base[a], _ = kdtree_rec_base.search_knn_vector_3d(centroid_loc_base[a], p_num)
            # test_time_end = time.time()
            fps_time_end= time.time()
            # log_string(LOG_FOUT, "kdtree method time: %f" % (fps_time_end - fps_time_begin))
            print('k nearest neighbor done...')

            
            c1_ind = torch.zeros(pointNum)
            # new_color1 = torch.zeros(pointNum, 3, dtype=torch.float32).to(device)
            new_color1 = torch.zeros(pointNum, dtype=torch.float32).to(device)
            new_color2 = torch.zeros(pointNum, dtype=torch.float32).to(device)
            new_color3 = torch.zeros(pointNum, dtype=torch.float32).to(device)
            datas1 = torch.zeros((args.test_batch_size, SEQ_LEN, 4, p_num)).to(device)
            datas2 = torch.zeros((args.test_batch_size, SEQ_LEN, 4, p_num)).to(device)
            datas3 = torch.zeros((args.test_batch_size, SEQ_LEN, 4, p_num)).to(device)
            ji = 0
            process_time_sum = 0
            cube_search_time_sum = 0
            model_time_sum = 0
            # test_time_start2 = time.time()
            if search:
                # group_idx = torch.zeros(numPatch, p_num).to(device)
                group_idx =torch.tensor( idxnn_rec_base).to(device)
            else:
                group_idx = torch.tensor(idxnn_rec_base).to(device)
            
            idxnn_rec = np.zeros((centroid_loc_base.shape[0], p_num), dtype=int)

            process_one_pc_time_begin = time.time()
            for ii in tqdm(range(numPatch), desc=lines[i+enhance_frame]):
                process_time1 = time.time()
                # print("patch_test: %d / %d" % (ii+1, numPatch))
                curPatchIdx_rec_base = idxnn_rec_base[ii, :]
                if search:
                    curPatchLoc_rec_base = rec_loc[base_frame][curPatchIdx_rec_base,:]
                    curPatchCol_rec_base = rec_color[base_frame][curPatchIdx_rec_base,:]
                else:
                    curPatchLoc_rec_base = rec_loc[enhance_frame][curPatchIdx_rec_base,:]
                    curPatchCol_rec_base = rec_color[enhance_frame][curPatchIdx_rec_base,:]
                # pt1 = loadPointCloud(curPatchLoc_rec_base, curPatchCol_rec_base)

                # data_id = group_idx[ii, :].long()
                
                for jj in range(SEQ_LEN):
                    if search and jj == base_frame:
                        continue
                    

                    if search:
                        cube_search_time_begin = time.time()
                        idxnn_rec[ii] = cube_search(kdtree_rec_base, kdtree_rec[jj], curPatchLoc_rec_base, curPatchCol_rec_base, 
                                              rec_loc, rec_color, ori_color, centroid_loc_base[ii], p_num,  None, ii, jj, step_size=1.2)
                        cube_search_time_end = time.time()
                        cube_search_time_sum += (cube_search_time_end - cube_search_time_begin)
                        # print(f"cube_search time: {cube_search_time_end - cube_search_time_begin}")
                        # if jj == enhance_frame:
                        #     group_idx[ii, :] =torch.tensor( idxnn_rec[ii]).to(device)
                        
                    else:
                         _, idxnn_rec[ii], _ = kdtree_rec[jj].search_knn_vector_3d(centroid_loc_base[ii], p_num)
                    

                    
                    curPatchIdx_rec = idxnn_rec[ii, :]
                    best_curPatchLoc_rec = rec_loc[jj][curPatchIdx_rec, :]
                    best_curPatchCol_rec = rec_color[jj][curPatchIdx_rec, :]


                    if search:
                        data_loc_base = torch.tensor(curPatchLoc_rec_base).to(device)
                        data_col_base = torch.tensor(curPatchCol_rec_base).to(device)
                    
                    data_loc = torch.tensor(best_curPatchLoc_rec).to(device)
                    data_col = torch.tensor(best_curPatchCol_rec).to(device)
                    
                    if normalize:
                        data_col[:, 0] = data_col[:, 0] / 255
                        data_col[:, 1] = (data_col[:, 1] - 0.5) / 255
                        data_col[:, 2] = (data_col[:, 2] - 0.5) / 255
                    
                    if search:
                        data1_base = torch.cat((data_loc_base, torch.unsqueeze(data_col_base[:, 0], dim=-1)), dim=-1)
                        data1_base = torch.unsqueeze(data1_base, dim=0)
                        data1_base = torch.unsqueeze(data1_base, dim=0)
                        data1_base = data1_base.permute(0, 1, 3, 2)
                        datas1[ji, base_frame, :, :] = data1_base
                    data1 = torch.cat((data_loc, torch.unsqueeze(data_col[:, 0], dim=-1)), dim=-1)
                    data1 = torch.unsqueeze(data1, dim=0)
                    data1 = torch.unsqueeze(data1, dim=0) # [1, 1, 1024, 4]
                    data1 = data1.permute(0, 1, 3, 2) # [1, 1, 4, 1024] [batchsize, seq_len, channels, pts]
                    datas1[ji, jj, :, :] = data1


                    if search:
                        data2_base = torch.cat((data_loc_base, torch.unsqueeze(data_col_base[:, 1], dim=-1)), dim=-1)
                        data2_base = torch.unsqueeze(data2_base, dim=0)
                        data2_base = torch.unsqueeze(data2_base, dim=0)
                        data2_base = data2_base.permute(0, 1, 3, 2)
                        datas2[ji, base_frame, :, :] = data2_base
                    data2 = torch.cat((data_loc, torch.unsqueeze(data_col[:, 1], dim=-1)), dim=-1)
                    data2 = torch.unsqueeze(data2, dim=0)
                    data2 = torch.unsqueeze(data2, dim=0)
                    data2 = data2.permute(0, 1, 3, 2)
                    datas2[ji, jj, :, :] = data2


                    if search:
                        data3_base = torch.cat((data_loc_base, torch.unsqueeze(data_col_base[:, 2], dim=-1)), dim=-1)
                        data3_base = torch.unsqueeze(data3_base, dim=0)
                        data3_base = torch.unsqueeze(data3_base, dim=0)
                        data3_base = data3_base.permute(0, 1, 3, 2)
                        datas3[ji, base_frame, :, :] = data3_base
                    data3 = torch.cat((data_loc, torch.unsqueeze(data_col[:, 2], dim=-1)), dim=-1)
                    data3 = torch.unsqueeze(data3, dim=0)
                    data3 = torch.unsqueeze(data3, dim=0)
                    data3 = data3.permute(0, 1, 3, 2)
                    datas3[ji, jj, :, :] = data3
                    # ****************************** #结束# 循环处理一个序列内的seq_len帧 ******************************

                
                ji = ji + 1
                if ji < args.test_batch_size:
                    continue
                ji = 0

                model_time_begin = time.time()
                with torch.no_grad():
                    logits1 = model1(datas1)
                    logits2 = model2(datas2)
                    logits3 = model3(datas3)

                logits1 = torch.squeeze(logits1)
                logits2 = torch.squeeze(logits2)
                logits3 = torch.squeeze(logits3)
                process_time2 = time.time()
                process_time_sum += (process_time2 - process_time1)
                # print(f"process one patch time: {process_time2 - process_time1}")


                for kk in range(args.test_batch_size):
                    idx_temps = group_idx[ii - args.test_batch_size + 1 + kk, :].long()
                    logit1 = logits1[kk, :]
                    logit2 = logits2[kk, :]
                    logit3 = logits3[kk, :]
                    for t, m in enumerate(idx_temps):
                        new_color1[m] += logit1[t]
                        new_color2[m] += logit2[t]
                        new_color3[m] += logit3[t]
                        c1_ind[m] += 1
                model_time_end = time.time()
                model_time_sum += (model_time_end - model_time_begin)

            process_one_pc_time_end = time.time()

            print("patch_test done...")
            patch_fuse_time_beg = time.time()

            rec_color_ten = torch.tensor(rec_color[enhance_frame])
            if normalize:
                rec_color_ten[:, 0] = rec_color_ten[:, 0] / 255
                rec_color_ten[:, 1] = (rec_color_ten[:, 1] - 0.5) / 255
                rec_color_ten[:, 2] = (rec_color_ten[:, 2] - 0.5) / 255

            for pn in range(pointNum):
                # 未在任何patch中出现，保持原始颜色
                if c1_ind[pn] == 0:
                    # new_color1[pn] = rec_color_ten[pn]
                    new_color1[pn] = rec_color_ten[pn, 0]
                    new_color2[pn] = rec_color_ten[pn, 1]
                    new_color3[pn] = rec_color_ten[pn, 2]
                # 在多个patch中取平均值
                elif c1_ind[pn] > 1:
                    new_color1[pn] /= c1_ind[pn]
                    new_color2[pn] /= c1_ind[pn]
                    new_color3[pn] /= c1_ind[pn]


            patch_fuse_time = time.time()

            # output_color11 = np.array(new_color1.cpu())
            output_color11 = np.array(torch.unsqueeze(new_color1, dim=-1).cpu())
            output_color12 = np.array(torch.unsqueeze(new_color2, dim=-1).cpu())
            output_color13 = np.array(torch.unsqueeze(new_color3, dim=-1).cpu())
            
            if normalize:
                # yuv反归一化
                output_color11 = output_color11 * 255
                output_color12 = output_color12 * 255 + 0.5
                output_color13 = output_color13 * 255 + 0.5

            # output_color = output_color11 # yuv一起
            output_color = np.concatenate((output_color11, output_color12, output_color13), axis=-1) # yuv分开
            output_color = np.clip(np.round(yuv2rgb(output_color)), 0, 255)
            output = pointcloud_rec[enhance_frame]

            
            # psnr_ori1 = psnr(ori_color[1], rec_color[1], MAX=255)
            # psnr_pred1 = psnr(ori_color[1], rgb2yuv(output_color), MAX=255)

            psnr_ori1 = psnr(ori_color[enhance_frame][:, 0], rec_color[enhance_frame][:, 0], MAX=255)
            psnr_pred1 = psnr(ori_color[enhance_frame][:, 0], rgb2yuv(output_color)[:, 0], MAX=255)
            
            psnr_ori2 = psnr(ori_color[enhance_frame][:, 1], rec_color[enhance_frame][:, 1], MAX=255)
            psnr_pred2 = psnr(ori_color[enhance_frame][:, 1], rgb2yuv(output_color)[:, 1], MAX=255)
            
            psnr_ori3 = psnr(ori_color[enhance_frame][:, 2], rec_color[enhance_frame][:, 2], MAX=255)
            psnr_pred3 = psnr(ori_color[enhance_frame][:, 2], rgb2yuv(output_color)[:, 2], MAX=255)

            psnr_ori = psnr(ori_color[enhance_frame], rec_color[enhance_frame], MAX=255)
            psnr_pred = psnr(ori_color[enhance_frame], rgb2yuv(output_color), MAX=255)

            log_string(LOG_FOUT, "psnr_y for original:  %f" % psnr_ori1)
            log_string(LOG_FOUT, "psnr_y for pred:  %f" % psnr_pred1)           
            log_string(LOG_FOUT, "psnr_u for original:  %f" % psnr_ori2)
            log_string(LOG_FOUT, "psnr_u for pred:  %f" % psnr_pred2)
            log_string(LOG_FOUT, "psnr_v for original:  %f" % psnr_ori3)
            log_string(LOG_FOUT, "psnr_v for pred:  %f" % psnr_pred3)
            log_string(LOG_FOUT, "psnr_overall for original:  %f" % psnr_ori)
            log_string(LOG_FOUT, "psnr_overall for pred:  %f" % psnr_pred)
            
            output[:, 3:] = output_color
            filepath = os.path.join(args.pred_path, lines[i+enhance_frame])
            write_ply(output, filepath)


            # log_string(LOG_FOUT, "patch_seg time:  %f" % (patch_seg_time2 - patch_seg_time1))
            # total_time = (fps_time_end - fps_time_begin) + (process_time_sum) + (patch_fuse_time - patch_fuse_time_beg)
            total_time = (fps_time_end - fps_time_begin) + (process_one_pc_time_end - process_one_pc_time_begin) + (patch_fuse_time - patch_fuse_time_beg)
            sts_time = (fps_time_end - fps_time_begin) + ( cube_search_time_sum)
            log_string(LOG_FOUT, "total time:  %f" % total_time)
            log_string(LOG_FOUT, "processing time:  %f" % process_time_sum)
            log_string(LOG_FOUT, "fps+knn time:  %f %.2f%%" % (fps_time_end - fps_time_begin, (fps_time_end - fps_time_begin) / total_time * 100))
            log_string(LOG_FOUT, "sts time:  %f %.2f%%" % (sts_time, sts_time / total_time * 100))
            log_string(LOG_FOUT, "model process time:  %f %.2f%%" % (model_time_sum, model_time_sum / total_time * 100))
            log_string(LOG_FOUT, "patch_fuse time:  %f %.2f%%" % ((patch_fuse_time - patch_fuse_time_beg), (patch_fuse_time - patch_fuse_time_beg) / total_time * 100))

        # t22 = time.time()
        # log_string(LOG_FOUT, "去噪一个点云total time:  %f" % (t22 - t11))
        if i + SEQ_LEN == total_lines:
            break

    LOG_FOUT.close()



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud quality Enhancement')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--pth_path', type=str, default='pths/final_2023')
    parser.add_argument('--log_path', type=str, default='logs/final_2023')
    parser.add_argument('--log_path_test', type=str, default='logs_test_2023/final_2023')
    parser.add_argument('--train_h5_txt', type=str, default='data_LT/h5/2048_2.0_predlift_seq3_cubesearch/trainFile.txt')
    parser.add_argument('--valid_h5_txt', type=str, default='data_LT/h5/2048_2.0_predlift_seq3_cubesearch/testFile.txt')
    parser.add_argument('--test_ply_txt', type=str, default='data_LT/data_ori_add/time/redandblack_r01.txt')
    parser.add_argument('--test_ori_ply', type=str, default='/home/lrx/pcc/mpeg-pcc-tmc2/data/8iVFBv2/redandblack/Ply')
    parser.add_argument('--test_rec_ply', type=str, default='/home/lrx/pcc/mpeg-pcc-tmc13/rec/redandblack/predlift/r01/ply/')
    parser.add_argument('--train_channel', type=int, default=0, help='0:Y, 1:U, 2:V')
    
    parser.add_argument('--model', type=str, default='GQE-Net', metavar='N',
                        help='Model to use, GQE-Net')
    parser.add_argument('--dataset', type=str, default='WPCSD', metavar='N',
                        choices=['WPCSD'])
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='test_batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=180, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--bit_rate_point', type=str, default='r01_r06_yuv_GQE-Net')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--has_model', type=bool, default=False,
                        help='Checkpoints')
    parser.add_argument('--lr', type=float, default=0.0016, metavar='LR',
                        help='Learning rate (default: 0.005, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Enables CUDA training')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluate the model (Test stage)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to use')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model1_path', type=str, default='pths/final_2023/GQE-Net/2024-08-31/y/model_16.pth', metavar='N',
                        help='Pretrained model1 path')
    parser.add_argument('--model2_path', type=str, default='pths/final_2023/GQE-Net/2024-08-31/u/model_20.pth', metavar='N',
                        help='Pretrained model2 path')
    parser.add_argument('--model3_path', type=str, default='pths/final_2023/GQE-Net/2024-08-31/v/model_18.pth', metavar='N',
                        help='Pretrained model3 path')
    parser.add_argument('--pred_path', type=str, default='pths/final_2023/GQE-Net/2024-08-31/model_16_20_18')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(devices + args.gpu)
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')

        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
