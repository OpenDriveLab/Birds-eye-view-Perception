"""
可视化sample累积的背景occ+前景occ
"""
import numpy as np
from mayavi import mlab
import os 
import mmcv
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

num_classes = 17
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
occupancy_size = [0.2, 0.2, 0.2]
voxel_size = 0.2

occ_xdim = int((point_cloud_range[3] - point_cloud_range[0]) / occupancy_size[0])
occ_ydim = int((point_cloud_range[4] - point_cloud_range[1]) / occupancy_size[1])
occ_zdim = int((point_cloud_range[5] - point_cloud_range[2]) / occupancy_size[2])
voxel_num = occ_xdim*occ_ydim*occ_zdim
add_ego_car = True

def generate_the_ego_car():
    ego_range = [-2, -1, -1.5, 2, 1, 0]
    ego_voxel_size=occupancy_size
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_x, ego_point_y, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*num_classes).astype(np.uint8)
    ego_points_flow = np.zeros((ego_point_xyz.shape[0], 2))
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    ego_dict['flow'] = ego_points_flow  

    return ego_dict

def obtain_points_label(occ):
    occ_index, occ_cls = occ[:, 0], occ[:, 1]
    occ = np.ones(voxel_num, dtype=np.int8)*11
    occ[occ_index[:]] = occ_cls  # (voxel_num)
    points = []
    for i in range(len(occ_index)):
        indice = occ_index[i]
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_xdim
        z = indice // (occ_xdim*occ_xdim)
        point_x = (x + 0.5) / occ_xdim * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        point_y = (y + 0.5) / occ_ydim * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        point_z = (z + 0.5) / occ_zdim * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points.append([point_x, point_y, point_z])
    
    points = np.stack(points)
    points_label = occ_cls


    return points, points_label

def get_box_corners(box):
    xc, yc, zc, length, width, height, theta = box  # theta是绕z轴逆时针旋转的角度, 逆时针为正，与+x轴重合角度为0
    # 点的命名顺序如下standard_bottom：0-1-2-3
    standard_bottom = np.array([[-length/2, width/2], [-length/2, -width/2], [length/2, -width/2], [length/2, width/2]])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    bottom = np.dot(rotation_matrix, standard_bottom.T).T  # (4, 2)
    corner_points = np.array([[xc+bottom[0][0], yc+bottom[0][1], zc - height/2],
                              [xc+bottom[1][0], yc+bottom[1][1], zc - height/2],
                              [xc+bottom[2][0], yc+bottom[2][1], zc - height/2],
                              [xc+bottom[3][0], yc+bottom[3][1], zc - height/2],
                              [xc+bottom[0][0], yc+bottom[0][1], zc + height/2],
                              [xc+bottom[1][0], yc+bottom[1][1], zc + height/2],
                              [xc+bottom[2][0], yc+bottom[2][1], zc + height/2],
                              [xc+bottom[3][0], yc+bottom[3][1], zc + height/2]])

    return corner_points

def plot3dbox(corner_points):
    idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
    x = corner_points[idx, 0]
    y = corner_points[idx, 1]
    z = corner_points[idx, 2]
    mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', 
                representation='wireframe', line_width=3)

def visualize_lidar(points, labels, ego_dict, result_box=None):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    colors_map = np.array(
        [   
            [255, 158, 0, 255],  #  1 car  orange
            [255, 99, 71, 255],  #  2 truck  Tomato
            [255, 140, 0, 255],  #  3 trailer  Darkorange
            [255, 69, 0, 255],  #  4 bus  Orangered
            [233, 150, 70, 255],  #  5 construction_vehicle  Darksalmon
            [220, 20, 60, 255],  #  6 bicycle  Crimson
            [255, 61, 99, 255],  #  7 motorcycle  Red
            [0, 0, 230, 255],  #  8 pedestrian  Blue
            [47, 79, 79, 255],  #  9 traffic_cone  Darkslategrey
            [112, 128, 144, 255],  #  10 barrier  Slategrey
            [0, 207, 191, 255],  # 11  driveable_surface  nuTonomy green  
            [175, 0, 75, 255],  #  12 other_flat  
            [75, 0, 75, 255],  #  13  sidewalk 
            [112, 180, 60, 255],  # 14 terrain  
            [222, 184, 135, 255], # 15 manmade Burlywood 
            [0, 175, 0, 255],  # 16 vegetation  Green
            [0, 0, 0, 255],  # unknown
        ]
    ).astype(np.uint8)

    point_colors = np.zeros(points.shape[0])
    for cls_index in range(num_classes):
        class_point = labels == cls_index
        point_colors[class_point] = cls_index+1 

    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    lidar_plot = mlab.points3d(x, y, z, point_colors,
                                scale_factor=1.1*voxel_size,
                                mode="cube",
                                scale_mode = "vector",
                                opacity=1.0,
                                vmin=1,
                                vmax=17,
                                )
    lidar_plot.module_manager.scalar_lut_manager.lut.table = colors_map

    if add_ego_car:
        ego_point_xyz = ego_dict['point']
        ego_points_label = ego_dict['label']
        ego_points_flow = ego_dict['flow']

        ego_color = np.linalg.norm(ego_point_xyz, axis=-1)
        ego_color = ego_color / ego_color.max()

        ego_plot = mlab.points3d(ego_point_xyz[:, 0], ego_point_xyz[:, 1], ego_point_xyz[:, 2], 
                                ego_color, 
                                colormap="rainbow",
                                scale_factor=0.95*voxel_size,
                                mode="cube",
                                opacity=1.0,
                                scale_mode='none',
                                )


    if result_box:
        gt_bboxes_3d = result_box['gt_bboxes_3d']
        gt_labels_3d = result_box['gt_labels_3d']

        for i in range(len(gt_bboxes_3d)):
            box = gt_bboxes_3d[i]
            if gt_labels_3d[i] == -1:
                continue
            corner_points = get_box_corners(box)
            plot3dbox(corner_points)


    # back_view
    view_type ='back_view'
    if view_type =='back_view':
        scene = figure
        scene.scene.z_plus_view()
        scene.scene.camera.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
        scene.scene.camera.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
        scene.scene.camera.clipping_range = [0.18978054185107493, 189.78054185107493]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

    mlab.show()


root_dir = 'data/test_data'


index = 20
points_path = os.path.join(root_dir, '{:03d}_occ.npy'.format(index))


occ = np.load(points_path) # (N, 2)
points, labels = obtain_points_label(occ)

print(np.unique(labels))

ego_dict = generate_the_ego_car()
visualize_lidar(points, labels, ego_dict)



        

    

