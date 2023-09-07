import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from meshnet.data.dataset import NodeType

def node_type(label: str) -> int:
        if label == 'INFLOW':
            return NodeType.INFLOW
        elif label == 'OUTFLOW':
            return NodeType.OUTFLOW
        elif label == 'WALL_BOUNDARY':
            return NodeType.WALL_BOUNDARY
        elif label == 'OBSTACLE':
            return NodeType.OBSTACLE
        else:
            return NodeType.NORMAL

def process_file_2d(
        config: dict,
) -> Data:
    with open(osp.join(config['predict_dir'], 'data', 'cad_{:03d}.geo'.format(config['name'])), 'r') as f:
        # read lines and remove comments
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

        # extract points, lines and circles
        points = [line for line in lines if line.startswith('Point')]
        lines__ = [line for line in lines if line.startswith('Line')]
        circles = [line for line in lines if line.startswith('Ellipse')]
        extrudes = [line for line in lines if line.startswith('Extrude')]
        physical_curves = [line for line in lines if line.startswith('Physical Curve')]

        # extract coordinates and mesh size
        points_id = torch.Tensor([int(line.split('(')[1].split(')')[0]) for line in points]).long()
        _, indices = torch.sort(points_id)
        points = torch.Tensor([[float(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in points])
        points = points[indices]
        points = points[:, :-1]

        # extract edges
        lines_id = torch.Tensor([int(line.split('(')[1].split(')')[0]) for line in lines__]).long()
        lines__ = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in lines__]).long()
        circles_id = torch.Tensor([int(line.split('(')[1].split(')')[0]) for line in circles]).long()
        circles = torch.Tensor([[int(p) for p in line.split('{')[1].split('}')[0].split(',')] for line in circles]).long()[:,[0,2]]
        edges_id = torch.cat([lines_id, circles_id], dim=0) - 1
        _, indices = torch.sort(edges_id)
        edges = torch.cat([lines__, circles], dim=0)-1
        edges = edges[indices]

        # add extruded points and edges
        for extrude in extrudes:
            z_extrude = float(extrude.split('}')[0].split(',')[-1])
            extruded_curves_id = torch.Tensor([int(extrude.split('{')[3:][i].split('}')[0]) for i in range(len(extrude.split('{')[3:]))]).long() - 1
            extruded_points_id = []
            new_extruded_points_id = []
            for id in extruded_curves_id:
                for i in edges[id]:
                    if not i in extruded_points_id:
                        extruded_points_id.append(i)
                        new_extruded_points_id.append(len(points))
                        points = torch.cat([points, torch.Tensor([points[i,0], points[i,1], z_extrude]).unsqueeze(0)], dim=0)
                        y = torch.cat([y, torch.Tensor([y[i]])], dim=0)
            extruded_points_id = torch.Tensor(extruded_points_id).long()
            new_extruded_points_id = torch.Tensor(new_extruded_points_id).long()

            new_extruded_curves = edges[extruded_curves_id]
            for i in range(len(extruded_points_id)):
                new_extruded_curves = torch.where(new_extruded_curves == extruded_points_id[i], new_extruded_points_id[i], new_extruded_curves)
            extruded_connexion = torch.cat([extruded_points_id.unsqueeze(dim=1), new_extruded_points_id.unsqueeze(dim=1)], dim=1)
            edges = torch.cat([edges, extruded_connexion, new_extruded_curves], dim=0)

        count = 0
        for i in range(points.shape[0]):
            if not (i-count) in edges:
                points = torch.cat([points[:i-count], points[i-count+1:]], dim=0)
                y = torch.cat([y[:i-count], y[i-count+1:]], dim=0)
                edges = edges - 1*(edges>(i-count))
                count += 1

        receivers = torch.min(edges, dim=1).values
        senders = torch.max(edges, dim=1).values
        packed_edges = torch.stack([senders, receivers], dim=1)
        # remove duplicates and unpack
        unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
        senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
        # create two-way connectivity
        edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

        # extract node types
        edge_types = torch.zeros(edges.shape[0], dtype=torch.long)
        for curve in physical_curves:
            label = curve.split('(')[1].split('"')[1]
            lines = curve.split('{')[1].split('}')[0].split(',')
            for line in lines:
                edge_types[int(line)-1] = node_type(label)
        tmp = torch.zeros(edges.shape[0], dtype=torch.long)
        for i in range(len(permutation)):
            tmp[permutation[i]] = edge_types[i]
        edge_types = torch.cat((tmp, tmp), dim=0)
        edge_types_one_hot = torch.nn.functional.one_hot(edge_types.long(), num_classes=NodeType.SIZE)

        # get edge attributes
        u_i = points[edge_index[0]][:,:2]
        u_j = points[edge_index[1]][:,:2]
        u_ij = torch.Tensor(u_i - u_j)
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
        edge_attr = torch.cat((u_ij, u_ij_norm, edge_types_one_hot),dim=-1).type(torch.float)

        # get node attributes
        x = torch.zeros(points.shape[0], NodeType.SIZE)
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                x[edge_index[i,j], edge_types[j]] = 1.0

        return Data(
            x=x.to(config['device']),
            edge_index=edge_index.to(config['device']),
            edge_attr=edge_attr.to(config['device']),
            name=torch.Tensor([config['name']]).long().to(config['device'])
        )

def process_file_3d(
        config: dict
) -> None:
    with open(osp.join(config['predict_dir'], 'data', 'cad_{:03d}.geo'.format(config['name'])), 'r') as f:
        # read lines and remove comments
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if not (line.startswith('//') or line.startswith('SetFactory'))]

        # extract geometries
        box = [line for line in lines if line.startswith('Box')]
        cylinders = [line for line in lines if line.startswith('Cylinder')]

        # Infer number of points and edges
        n_points = 8 + 6*len(cylinders)
        points = torch.zeros(n_points, 3)

        n_edges = 12 + 9*len(cylinders)
        edges = torch.zeros(n_edges, 2, dtype=torch.long)

        # extract coordinates from geometries
        box = [float(box[0].split('{')[1].split('}')[0].split(', ')[i]) for i in range(len(box[0].split('{')[1].split('}')[0].split(', ')))] # [xs, ys, zs, dx, dy, dz]
        cylinders = [[float(cylinders[j].split('{')[1].split('}')[0].split(', ')[i]) for i in range(len(cylinders[0].split('{')[1].split('}')[0].split(', '))-1)] for j in range(len(cylinders))] # [xs, ys, zs, dx, dy, dz, r]

        # extract mesh sizes
        l = float([line for line in lines if line.startswith('l')][0].split(' ')[-1].split(';')[0])
        c = [float([line for line in lines if line.startswith('c')][i].split(' ')[-1].split(';')[0]) for i in range(len(cylinders))]

        # create points, targets mesh size and edges for the box
        points[0] = torch.Tensor([box[0], box[1], box[2]])
        points[1] = torch.Tensor([box[0]+box[3], box[1], box[2]])
        points[2] = torch.Tensor([box[0]+box[3], box[1]+box[4], box[2]])
        points[3] = torch.Tensor([box[0], box[1]+box[4], box[2]])
        points[4] = torch.Tensor([box[0], box[1], box[2]+box[5]])
        points[5] = torch.Tensor([box[0]+box[3], box[1], box[2]+box[5]])
        points[6] = torch.Tensor([box[0]+box[3], box[1]+box[4], box[2]+box[5]])
        points[7] = torch.Tensor([box[0], box[1]+box[4], box[2]+box[5]])
        
        edges[0] = torch.Tensor([0, 1]).long()
        edges[1] = torch.Tensor([1, 2]).long()
        edges[2] = torch.Tensor([2, 3]).long()
        edges[3] = torch.Tensor([3, 0]).long()
        edges[4] = torch.Tensor([4, 5]).long()
        edges[5] = torch.Tensor([5, 6]).long()
        edges[6] = torch.Tensor([6, 7]).long()
        edges[7] = torch.Tensor([7, 4]).long()
        edges[8] = torch.Tensor([0, 4]).long()
        edges[9] = torch.Tensor([1, 5]).long()
        edges[10] = torch.Tensor([2, 6]).long()
        edges[11] = torch.Tensor([3, 7]).long()

        # create points and edges for the cylinders
        for i in range(len(cylinders)):
            points[8+6*i] = torch.Tensor([cylinders[i][0]+cylinders[i][-1], cylinders[i][1], cylinders[i][2]])
            points[8+6*i+1] = torch.Tensor([cylinders[i][0]+np.cos(2*np.pi/3)*cylinders[i][-1], cylinders[i][1]+np.sin(2*np.pi/3)*cylinders[i][-1], cylinders[i][2]])
            points[8+6*i+2] = torch.Tensor([cylinders[i][0]+np.cos(4*np.pi/3)*cylinders[i][-1], cylinders[i][1]+np.sin(4*np.pi/3)*cylinders[i][-1], cylinders[i][2]])
            points[8+6*i+3] = torch.Tensor([cylinders[i][0]+cylinders[i][-1], cylinders[i][1], cylinders[i][2]+cylinders[i][5]])
            points[8+6*i+4] = torch.Tensor([cylinders[i][0]+np.cos(2*np.pi/3)*cylinders[i][-1], cylinders[i][1]+np.sin(2*np.pi/3)*cylinders[i][-1], cylinders[i][2]+cylinders[i][5]])
            points[8+6*i+5] = torch.Tensor([cylinders[i][0]+np.cos(4*np.pi/3)*cylinders[i][-1], cylinders[i][1]+np.sin(4*np.pi/3)*cylinders[i][-1], cylinders[i][2]+cylinders[i][5]])
        
            edges[12+9*i] = torch.Tensor([8+6*i, 8+6*i+1]).long()
            edges[12+9*i+1] = torch.Tensor([8+6*i+1, 8+6*i+2]).long()
            edges[12+9*i+2] = torch.Tensor([8+6*i+2, 8+6*i]).long()
            edges[12+9*i+3] = torch.Tensor([8+6*i+3, 8+6*i+4]).long()
            edges[12+9*i+4] = torch.Tensor([8+6*i+4, 8+6*i+5]).long()
            edges[12+9*i+5] = torch.Tensor([8+6*i+5, 8+6*i+3]).long()
            edges[12+9*i+6] = torch.Tensor([8+6*i, 8+6*i+3]).long()
            edges[12+9*i+7] = torch.Tensor([8+6*i+1, 8+6*i+4]).long()
            edges[12+9*i+8] = torch.Tensor([8+6*i+2, 8+6*i+5]).long()

        receivers = torch.min(edges, dim=1).values
        senders = torch.max(edges, dim=1).values
        packed_edges = torch.stack([senders, receivers], dim=1)
        # remove duplicates and unpack
        unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
        senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
        # create two-way connectivity
        edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

        # extract edges labels
        edge_types = torch.zeros(edges.shape[0], dtype=torch.long)
        # inflow
        edge_types[3] = NodeType.INFLOW
        edge_types[7] = NodeType.INFLOW
        edge_types[8] = NodeType.INFLOW
        edge_types[11] = NodeType.INFLOW
        # outflow
        edge_types[1] = NodeType.OUTFLOW
        edge_types[5] = NodeType.OUTFLOW
        edge_types[9] = NodeType.OUTFLOW
        edge_types[10] = NodeType.OUTFLOW
        # walls
        edge_types[0] = NodeType.WALL_BOUNDARY
        edge_types[2] = NodeType.WALL_BOUNDARY
        edge_types[4] = NodeType.WALL_BOUNDARY
        edge_types[6] = NodeType.WALL_BOUNDARY
        # obstacles
        edge_types[12:] += NodeType.OBSTACLE
        
        # convert edges labels to edge_index format
        tmp = torch.zeros(edges.shape[0], dtype=torch.long)
        for i in range(len(permutation)):
            tmp[permutation[i]] = edge_types[i]
        edge_types = torch.cat((tmp, tmp), dim=0)

        # convert edge labels to one-hot vector
        edge_types_one_hot = torch.nn.functional.one_hot(edge_types.long(), num_classes=NodeType.SIZE)

        # construct edge attributes
        u_i = points[edge_index[0]][:,:2]
        u_j = points[edge_index[1]][:,:2]
        u_ij = torch.Tensor(u_i - u_j)
        u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
        edge_attr = torch.cat((u_ij, u_ij_norm, edge_types_one_hot),dim=-1).type(torch.float)

        # get node labels
        x = torch.zeros(points.shape[0], NodeType.SIZE)
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                x[edge_index[i,j], edge_types[j]] = 1.0

        return Data(
            x=x.to(config['device']),
            edge_index=edge_index.to(config['device']),
            edge_attr=edge_attr.to(config['device']),
            name=torch.Tensor([config['name']]).long().to(config['device'])
        )

def triangles_to_edges(dim: int, faces: torch.Tensor) -> torch.Tensor:
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    if (dim == 2):
        edges = torch.vstack((faces[:, 0:2],
                            faces[:, 1:3],
                            torch.hstack((faces[:, 2].unsqueeze(dim=-1),
                                            faces[:, 0].unsqueeze(dim=-1)))
                            ))
    elif (dim == 3):
        edges = torch.vstack((faces[:, 0:2],
                            faces[:, 1:3],
                            faces[:, 2:4],
                            torch.hstack((faces[:, 3].unsqueeze(dim=-1),
                                            faces[:, 0].unsqueeze(dim=-1)))
                            ))
    else:
        raise ValueError("The dimension must be either 2 or 3.")
    receivers = torch.min(edges, dim=1).values
    senders = torch.max(edges, dim=1).values
    packed_edges = torch.stack([senders, receivers], dim=1)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

def write_field(path:str, field: torch.Tensor, name: str) -> None:
    with open(osp.join(path, f'{name}.txt'), 'w') as f:
        f.write(f'{len(field)}\t\n')
        for i in range(0, len(field), 5):
            if (i+5>len(field)):
                r = len(field) - i
                if r == 1:
                    f.write(f'\t{field[i]}\n')
                elif r == 2:
                    f.write(f'\t{field[i]}\t{field[i+1]}\n')
                elif r == 3:
                    f.write(f'\t{field[i]}\t{field[i+1]}\t{field[i+2]}\n')
                elif r == 4:
                    f.write(f'\t{field[i]}\t{field[i+1]}\t{field[i+2]}\t{field[i+3]}\n')
            else:
                f.write(f'\t{field[i]}\t{field[i+1]}\t{field[i+2]}\t{field[i+3]}\t{field[i+4]}\n')