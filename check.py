import json 
from tqdm import tqdm 
import numpy as np 
import torch
from torch.utils.data.dataloader import default_collate

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os 
import sys
import sys
sys.path.append('/home/malab/Desktop/Research/FINAL')

from base import _C


from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def adjacent_matrix(n, edges, device):
        mat = torch.zeros(n+1,n+1,dtype=torch.bool,device=device)
        if edges.size(0)>0:
            mat[edges[:,0], edges[:,1]] = 1
            mat[edges[:,1], edges[:,0]] = 1
        return mat

currentPath = os.getcwd()
relativePath = "data/wireframe/train.json"  #"../../../data/wireframe/train.json"
absolutePath = os.path.join(currentPath, relativePath)
imagePath = os.path.join(currentPath, "data/wireframe/images/00052226.png")

f = open(absolutePath, "r")


ann = json.load(f)
print(type(ann))
print('--------------------------------')

device = torch.device('cuda')

def select_data (ann, filename):
    for idx in range(len(ann)):
        if ann[idx]['filename'] == filename:
            return ann[idx]
        
ann = select_data(ann, '00052226.png')

junctions = torch.tensor(ann['junctions'], dtype=torch.float32)
junctions = junctions.to(device)



height, width = ann['height'], ann['width']

print('height:{} and width:{}'.format(height, width))

jmap = np.zeros((height,width),dtype=np.float32)
joff =np.zeros((2,height, width), dtype=np.float32)


print(f'jmap:{jmap}')
print(f'joff:{joff}')
print(f'shape of jmap: {jmap.shape} and shape of joff: {joff.shape} ')

junctions_np = junctions.cpu().numpy()
print(junctions_np.shape)
xint, yint = junctions_np[:,0].astype(np.int_), junctions_np[:,1].astype(np.int_)
#print (xint, yint)

off_x = junctions_np[:,0] - np.floor(junctions_np[:,0]) - 0.5
off_y = junctions_np[:,1] - np.floor(junctions_np[:,1]) - 0.5



jmap[yint, xint] = 1
joff[0,yint, xint] = off_x
joff[1,yint, xint] = off_y



jmap = torch.from_numpy(jmap).to(device)


edges_positive = torch.tensor(ann['edges_positive']) 
# edges_negative = torch.tensor(ann['edges_negative'])


pos_mat = adjacent_matrix(junctions.size(0),edges_positive, device)
# neg_mat = adjacent_matrix(junctions.size(0),edges_negative, device) 

line_annotations = torch.tensor(ann["line_annotations"], dtype=torch.float32, device=device)

print(f'line annotations: {len(line_annotations)}')

lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
# lines_neg = torch.cat((junctions[edges_negative[:2000,0]],junctions[edges_negative[:2000,1]]),dim=-1)

print(f'lines: {lines} \n and line shape:{lines.shape} ')

lmap, annotation, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0), line_annotations)


# print(f'line map: {lmap.shape}')
# print(f'line map: {lmap[6]}')

dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)[None]

hafm_dis = dismap.clamp(max=25)/25

def _normalize(inp):
    mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
    return inp/(mag+1e-6)

md_map = _normalize(lmap[:2])
st_map = _normalize(lmap[2:4])
ed_map = _normalize(lmap[4:])

md_ = md_map.reshape(2,-1).t()
st_ = st_map.reshape(2,-1).t()
ed_ = ed_map.reshape(2,-1).t()
Rt = torch.cat(
        (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
            torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
R = torch.cat(
        (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
            torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)

Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
pos_ = Rtst_.clone()
neg_ = Rted_.clone()
temp = pos_[:,swap_mask]
pos_[:,swap_mask] = neg_[:,swap_mask]
neg_[:,swap_mask] = temp

pos_[0] = pos_[0]#.clamp(min=1e-9)
pos_[1] = pos_[1]#.clamp(min=1e-9)
neg_[0] = neg_[0]#.clamp(min=1e-9)
neg_[1] = neg_[1]#.clamp(max=-1e-9)

mask = (dismap.view(-1)).float()

pos_map = pos_.reshape(-1,height,width)
neg_map = neg_.reshape(-1,height,width)

md_angle  = torch.atan2(md_map[1], md_map[0])
pos_angle = torch.atan2(pos_map[1],pos_map[0])
neg_angle = torch.atan2(neg_map[1],neg_map[0])
mask *= (pos_angle.reshape(-1)>np.pi/2.0)
mask *= (neg_angle.reshape(-1)<np.pi/2.0)

pos_angle_n = pos_angle/(np.pi/2)
neg_angle_n = -neg_angle/(np.pi/2)
md_angle_n  = md_angle/(np.pi*2) + 0.5
mask    = mask.reshape(height,width)


# import pdb; pdb.set_trace()
hafm_ang = torch.cat((md_angle_n[None],pos_angle_n[None],neg_angle_n[None],),dim=0)

print(f'annotation: {annotation - 1}')
print(annotation.shape)

# Define the custom colors in the order specified by the annotation map
custom_colors = ['#B91C1C',  # horizontalUpperEdge (0) - bg-red-400
                 '#34D399',  # wallEdge (1) - bg-green-400
                 '#60A5FA',  # horizontalLowerEdge (2) - bg-blue-400
                 '#FBBF24',  # doorEdge (3) - bg-yellow-400
                 '#C084FC',  # windowEdge (4) - bg-magenta-400
                 '#F97316']  # miscellaneousObjects (5) - bg-orange-600
custom_cmap = ListedColormap(custom_colors)

custom = ['#B91C1C',  # bg-red-400
          '#34D399',  # bg-green-400
          '#60A5FA',  # bg-blue-400
          '#FBBF24',  # bg-yellow-400
          '#C084FC',  # bg-magenta-400
          '#F97316']  # bg-orange-600

custom_cmap1 = ListedColormap(custom_colors)

def visualize_hafm(hafm_ang, hafm_dis, annotation, custom_cmap = None, custom_cmap1 = None, line_annotations = None , lines = None, image = None):
    # Assuming hafm_ang and hafm_dis are torch tensors, convert them to numpy for plotting
    hafm_ang_np = hafm_ang.cpu().detach().numpy()
    hafm_dis_np = hafm_dis.squeeze().cpu().numpy()
    annotation_np = annotation.squeeze().cpu().numpy()
    
    if lines is not None:
        lines_np = lines.cpu().detach().numpy()
    if line_annotations is not None:
        line_annotations_np = line_annotations.cpu().detach().numpy()

    fig, axs = plt.subplots(1, 1, figsize=(12, 7))
    # Visualize the anotation mask
    plt.imshow(annotation_np, cmap=custom_cmap, interpolation='bilinear')
    plt.title('Annotation Mask')
    plt.axis('off')

    plt.show()
    # # Visualize hafm_dis
    # axs[0].imshow(hafm_ang_np[0], cmap='viridis', interpolation='nearest')
    # axs[0].set_title('HAFM Rotational Angle Map')

    # axs[1].imshow(hafm_ang_np[1], cmap='viridis', interpolation='nearest')  # Using the first channel as an example
    # axs[1].set_title('HAFM Scaling Angle_1 Map')

    # axs[2].imshow(hafm_ang_np[2], cmap='viridis', interpolation='nearest')  # Using the first channel as an example
    # axs[2].set_title('HAFM Scaling Angle_2 Map')

    # axs[3].imshow(hafm_dis_np, cmap='hot', interpolation='nearest')
    # axs[3].set_title('HAFM Distance Map')
    
    # Visualize the image
    if image is not None:
        plt.imshow(image, cmap=custom_cmap, interpolation='bilinear')
    
    # Visualize the annotated lines
    if lines is not None and line_annotations is not None:
        for i in range(lines.shape[0]):
            line = lines_np[i]
            annotation_class = int(line_annotations_np[i])
            color = custom_colors[annotation_class]
            plt.plot([line[0], line[2]], [line[1], line[3]], color=color, linewidth=1.5)
            plt.scatter([line[0], line[2]], [line[1], line[3]], color=color, s=2, edgecolors='none', zorder=5)
    
    plt.title('Annotated Lines')
    plt.axis('off')
    plt.show()


    # Visualize the anotation mask
    plt.imshow(annotation_np, cmap=custom_cmap1, interpolation='bilinear')
    plt.title('Annotation Mask')
    plt.axis('off')

    plt.show()

data = Image.open(imagePath)
line_annotations = line_annotations - 1
annotation = annotation - 1
visualize_hafm(hafm_ang, hafm_dis, annotation, custom_cmap=custom_cmap, custom_cmap1=custom_cmap1, line_annotations=line_annotations, lines=lines, image=data)



data = np.array(data)

xx, yy = np.meshgrid(range(ann['width']),range(ann['height']))
im_tensor = torch.Tensor(data.transpose([2,0,1])).unsqueeze(0)
im_tensor = torch.nn.functional.interpolate(im_tensor, (ann['height'],ann['width']), mode='bilinear', align_corners=False)
afx = lmap[0].cpu().numpy() + xx
afy = lmap[1].cpu().numpy() + yy

image = im_tensor.squeeze().cpu().numpy().transpose([1,2,0])/ 255.0
plt.imshow(image)
plt.plot(afx, afy,'b', markersize=0.5)
plt.show()













    
