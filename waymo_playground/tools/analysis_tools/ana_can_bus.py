from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
import torch
import matplotlib.pyplot as plt

idx = 'b6c420c3a5bd4a219b1cb82ee5ea0aa7'
nusc_can = NuScenesCanBus(dataroot='data')
nusc = NuScenes(dataroot='data/nuscenes', version='v1.0-mini')

sample = nusc.get('sample', idx)
scene = nusc.get('scene', sample['scene_token'])
print(sample)

from collections import defaultdict


'''
rot = [[], [], [], []]
tran = [[],[], []]
for sample in nusc.sample:
    samp = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
    intrin = sens['camera_intrinsic']
    rota = sens['rotation']
    for i,each in enumerate(rota):        
        rot[i].append(each)
        
    for i, each in enumerate(sens['translation']):
        tran[i].append(each)

fig,axes = plt.subplots(2,4) 
ax1 = axes[0, 0]
ax1.hist(rot[0])
ax1.set_title('rot0')
ax2 = axes[0, 1]
ax2.hist(rot[1])
ax2.set_title('rot1')
ax3 = axes[0, 2]
ax3.hist(rot[2])
ax3.set_title('rot3')
ax4 = axes[0, 3]
ax4.hist(rot[3])
ax4.set_title('rot4')
ax5 = axes[1, 0]
ax5.hist(tran[0])
ax5.set_title('tran1')
ax6 = axes[1, 1]
ax6.hist(tran[1])
ax6.set_title('tran2')
ax7 = axes[1, 2]
ax7.hist(tran[2])
ax7.set_title('tran3')
plt.savefig('ext.png')

exit()
'''

#scene_name = 'scene-0061'
#pose = nusc_can.get_messages(scene_name, 'pose')
#rint(pose[0])
#scene = nusc.scene[3]
print(nusc.get( 'log', scene['log_token']))
sample_idx = scene['first_sample_token']
sample = nusc.get('sample', sample_idx)
i =0
while sample:
    print(sample['token'])
    lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose = nusc.get('ego_pose',lidar['ego_pose_token'])

    rotation = Quaternion(pose['rotation'])
    translation = pose['translation']
    #print(translation)

    patch_angle = quaternion_yaw(rotation) / np.pi *180
    print(i, translation, patch_angle)
    i+=1
    if sample['next'] == '':
        break
    sample = nusc.get('sample', sample['next'])

nusc_map_bos = NuScenesMap(dataroot='data/nuscenes', map_name=nusc.get( 'log', scene['log_token'])['location'])

ego_poses = nusc_map_bos.render_egoposes_on_fancy_map(nusc, scene_tokens=[scene['token']],out_path='tmp.', verbose=False)
