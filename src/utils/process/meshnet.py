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

def file(
        config: dict,
) -> Data:
    with open(osp.join(config['predict_dir'], 'data', 'cad_{:03d}'.format(config['name']), 'cad_{:03d}.geo_unrolled'.format(config['name'])), 'r') as f:
        # read lines and remove comments
        lines = f.readlines()
        lines = [line.replace(' ', '') for line in lines]

        # Extract mesh sizes variables
        mesh_sizes_variables = {}
        tmp = [line for line in lines if line.startswith("cl__")]
        for line in tmp:
            key = line.split('=')[0]
            value = float(line.split('=')[-1].split(';')[0])
            mesh_sizes_variables[key] = value
        
        # Extract coordinates and mesh sizes
        convert_points = {}
        coo = {}
        mesh_sizes = {}
        tmp = [line for line in lines if line.startswith("Point(")]
        i=0
        for line in tmp:
            key = line.split('(')[1].split(')')[0]
            convert_points[key] = i
            value = line.split('{')[1].split('}')[0].split(',')
            coo[key] = [float(value[i]) for i in range(3)]
            if (len(value)>3):
                mesh_sizes[key] = mesh_sizes_variables[value[-1]]
            i+=1
        points = torch.Tensor(list(coo.values()))

        # Convert mesh sizes to tensor
        y = torch.Tensor(list(mesh_sizes.values()))
        indices = torch.Tensor([convert_points[key] for key in mesh_sizes.keys()]).long()
        points = points[indices]

        # Extract edges
        edges = {}
        tmp = [line for line in lines if line.startswith("Line(") or line.startswith("Spline(")]
        n_cyl = int(len([line for line in lines if line.startswith("Spline(")])/(config['meshnet']['dim']-1))
        for line in tmp:
            key = line.split('(')[1].split(')')[0]
            value = line.split('{')[1].split('}')[0].split(',')
            edges[key] = value
        
        # Connectivity matrix
        convert_edges = {}
        connectivity = []
        for key, value in edges.items():
            convert_edges[key] = [len(connectivity)+i for i in range(len(value)-1)]
            for i in range(len(value)-1):
                connectivity.append([convert_points[value[i]], convert_points[value[i+1]]])
        edges = torch.Tensor(connectivity).long()

        # Identify edges to keep
        keep_edges = []
        for i in range(edges.shape[0]):
            if ((edges[i,0] in indices) and (edges[i,1] in indices)):
                keep_edges.append(i)
        edges = edges[keep_edges]

        # Add control edges
        for i in range(n_cyl):
            if (config['meshnet']['dim']==2):
                edges = torch.cat((edges, torch.Tensor([[4+i,4+i]]).long()))
            elif (config['meshnet']['dim']==3):
                edges = torch.cat((edges, torch.Tensor([[7+i,0]]).long()))
                edges = torch.cat((edges, torch.Tensor([[7+i,2]]).long()))
                edges = torch.cat((edges, torch.Tensor([[7+i,5]]).long()))
                edges = torch.cat((edges, torch.Tensor([[7+i,6]]).long()))

                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,1]]).long()))
                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,3]]).long()))
                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,4]]).long()))
                edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,7+n_cyl]]).long()))
            else:
                raise ValueError('Dimension not supported')

        receivers = torch.min(edges, dim=1).values
        senders = torch.max(edges, dim=1).values
        packed_edges = torch.stack([senders, receivers], dim=1)
        # Remove duplicates and unpack
        unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
        senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
        # Create two-way connectivity
        edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0).long()

        if (config['meshnet']['dim']==3):
            # Extract curves
            curves = {}
            tmp = [line for line in lines if line.startswith("CurveLoop(")]
            tmp = [line.replace('-', '') for line in tmp]
            for line in tmp:
                key = line.split('(')[1].split(')')[0]
                value = line.split('{')[1].split('}')[0].split(',')
                curves[key] = value

            # Extract surfaces
            surfaces_edges = {}
            tmp = [line for line in lines if line.startswith("PlaneSurface(") or line.startswith("Surface(")]
            for line in tmp:
                key = line.split('(')[1].split(')')[0]
                value = line.split('{')[1].split('}')[0].split(',')
                for i in range(len(value)):
                    if (i==0):
                        surfaces_edges[key] = curves[value[i]][:]
                    else:
                        surfaces_edges[key] += curves[value[i]][:]

        # Extract physical groups
        physical_groups = {}
        if (config['meshnet']['dim']==2):
            tmp = [line for line in lines if line.startswith("PhysicalCurve")]
        elif (config['meshnet']['dim']==3):
            tmp = [line for line in lines if line.startswith("PhysicalSurface")]
        else:
            raise ValueError('Dimension not supported')
        tmp = [line for line in lines if line.startswith("PhysicalSurface")]
        for line in tmp:
            key = line.split('"')[1].split('"')[0]
            value = line.split('{')[1].split('}')[0].split(',')
            for i in range(len(value)):
                if (config['meshnet']['dim']==2):
                    if (i==0):
                        physical_groups[key] = convert_edges[value[i]][:]
                    else:
                        physical_groups[key] += convert_edges[value[i]][:]
                elif (config['meshnet']['dim']==3):
                    for j in range(len(surfaces_edges[value[i]])):
                        if (i==0 and j==0):
                            physical_groups[key] = convert_edges[surfaces_edges[value[i]][j]][:]
                        else:
                            physical_groups[key] += convert_edges[surfaces_edges[value[i]][j]][:]
                else:
                    raise ValueError('Dimension not supported')

        # Extract edge physical groups
        if (config['meshnet']['dim']==2):
            physical_groups_order = ['WALL_BOUNDARY', 'OUTFLOW', 'INFLOW', 'OBSTACLE']
        elif (config['meshnet']['dim']==3):
            physical_groups_order = ['WALL_Y', 'WALL_Z', 'OUTFLOW', 'INFLOW', 'OBSTACLE']
        else:
            raise ValueError('Dimension not supported')
        physical_groups_edges = [0 for i in range(len(connectivity))]
        for key in physical_groups_order:
            for i in range(len(physical_groups[key])):
                if (key=='INFLOW'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.INFLOW
                elif (key=='OUTFLOW'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.OUTFLOW
                elif (key=='WALL' or key=='WALL_Y' or key=='WALL_Z'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.WALL_BOUNDARY
                elif (key=='OBSTACLE'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.OBSTACLE
                else:
                    raise ValueError('Physical group not recognized.')
        edge_types = torch.Tensor(physical_groups_edges).long().long()
        edge_types = edge_types[keep_edges]
        if (config['meshnet']['dim']==2):
            edge_types = torch.cat((edge_types, (NodeType.OBSTACLE*torch.ones(edges.shape[0]-edge_types.shape[0])).long().long()))
        elif (config['meshnet']['dim']==3):
            edge_types = torch.cat((edge_types, (NodeType.WALL_BOUNDARY*torch.ones(edges.shape[0]-edge_types.shape[0])).long().long()))
        else:
            raise ValueError('Dimension not supported')
        
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