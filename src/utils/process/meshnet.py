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

def file_2d(
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

def file_3d(
        config: dict
) -> None:
    with open(osp.join(config['predict_dir'], 'data', 'cad_{:03d}.geo_unrolled'.format(config['name'])), 'r') as f:
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
        n_cyl = int(len([line for line in lines if line.startswith("Spline(")])/2)
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
            edges = torch.cat((edges, torch.Tensor([[7+i,0]]).long()))
            edges = torch.cat((edges, torch.Tensor([[7+i,2]]).long()))
            edges = torch.cat((edges, torch.Tensor([[7+i,5]]).long()))
            edges = torch.cat((edges, torch.Tensor([[7+i,6]]).long()))

            edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,1]]).long()))
            edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,3]]).long()))
            edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,4]]).long()))
            edges = torch.cat((edges, torch.Tensor([[8+n_cyl+i,7+n_cyl]]).long()))  

        receivers = torch.min(edges, dim=1).values
        senders = torch.max(edges, dim=1).values
        packed_edges = torch.stack([senders, receivers], dim=1)
        # Remove duplicates and unpack
        unique_edges, permutation = torch.unique(packed_edges, return_inverse=True, dim=0)
        senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
        # Create two-way connectivity
        edge_index = torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0).long()

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
        tmp = [line for line in lines if line.startswith("PhysicalSurface")]
        for line in tmp:
            key = line.split('"')[1].split('"')[0]
            value = line.split('{')[1].split('}')[0].split(',')
            for i in range(len(value)):
                for j in range(len(surfaces_edges[value[i]])):
                    if (i==0 and j==0):
                        physical_groups[key] = convert_edges[surfaces_edges[value[i]][j]][:]
                    else:
                        physical_groups[key] += convert_edges[surfaces_edges[value[i]][j]][:]

        # Extract edge physical groups
        physical_groups_order = ['WALL_Y', 'WALL_Z', 'OUTFLOW', 'INFLOW', 'OBSTACLE']
        physical_groups_edges = [0 for i in range(len(connectivity))]
        for key in physical_groups_order:
            for i in range(len(physical_groups[key])):
                if (key=='INFLOW'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.INFLOW
                elif (key=='OUTFLOW'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.OUTFLOW
                elif (key=='WALL_Y' or key=='WALL_Z'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.WALL_BOUNDARY
                elif (key=='OBSTACLE'):
                    physical_groups_edges[physical_groups[key][i]] = NodeType.OBSTACLE
                else:
                    raise ValueError('Physical group not recognized.')
        edge_types = torch.Tensor(physical_groups_edges).long().long()
        edge_types = edge_types[keep_edges]
        edge_types = torch.cat((edge_types, (NodeType.WALL_BOUNDARY*torch.ones(edges.shape[0]-edge_types.shape[0])).long().long()))
        
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