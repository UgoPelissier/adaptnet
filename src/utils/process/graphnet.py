import torch
import meshio
import os.path as osp
from torch_geometric.data import Data
from graphnet.data.dataset import NodeType

def triangles_to_edges(faces: torch.Tensor) -> torch.Tensor:
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = torch.vstack((faces[:, 0:2],
                        faces[:, 1:3],
                        torch.hstack((faces[:, 2].unsqueeze(dim=-1),
                                        faces[:, 0].unsqueeze(dim=-1)))
                        ))
    receivers = torch.min(edges, dim=1).values
    senders = torch.max(edges, dim=1).values
    packed_edges = torch.stack([senders, receivers], dim=1)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

def tetra_to_edges(faces: torch.Tensor) -> torch.Tensor:
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = torch.vstack((faces[:, 0:2],
                        faces[:, 1:3],
                        faces[:, 2:4],
                        torch.hstack((faces[:, 3].unsqueeze(dim=-1),
                                        faces[:, 0].unsqueeze(dim=-1)))
                        ))
    receivers = torch.min(edges, dim=1).values
    senders = torch.max(edges, dim=1).values
    packed_edges = torch.stack([senders, receivers], dim=1)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    senders, receivers = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack([torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)], dim=0)

def vtk(
    config: dict
)->Data:
    mesh = meshio.read(osp.join(config['save_dir'], config['save_folder'], 'vtk', 'cad_{:03d}.vtk'.format(config["name"])))

    node_type = torch.zeros(mesh.points.shape[0])
    if (config['graphnet']['dim'] == 2):
        k = 1
    elif (config['graphnet']['dim'] == 3):
        k = 0
    else:
        raise ValueError("The dimension must be either 2 or 3.")
    for i in range(mesh.cells[k].data.shape[0]):
        for j in range(mesh.cells[k].data.shape[1]-1):
                node_type[mesh.cells[k].data[i,j]] = mesh.cell_data['CellEntityIds'][k][i][0] - 1

    node_type_one_hot = torch.nn.functional.one_hot(node_type.long(), num_classes=NodeType.SIZE)

    # get initial velocity
    v_0 = torch.zeros(mesh.points.shape[0], config['graphnet']['dim'])
    mask = (node_type.long())==torch.tensor(NodeType.INFLOW)
    if (config['graphnet']['dim'] == 2):
        v_0[mask] = torch.Tensor([config['graphnet']['u_0'], config['graphnet']['v_0']])
    elif (config['graphnet']['dim'] == 3):
        v_0[mask] = torch.Tensor([config['graphnet']['u_0'], config['graphnet']['v_0'], config['graphnet']['w_0']])
    else:
        raise ValueError("The dimension must be either 2 or 3.")

    # get features
    x = torch.cat((v_0, node_type_one_hot),dim=-1).type(torch.float)

    # get edge indices in COO format
    if (config['graphnet']['dim'] == 2):
        edge_index = triangles_to_edges(torch.Tensor(mesh.cells[1-k].data)).long()
    elif (config['graphnet']['dim'] == 3):
        edge_index = tetra_to_edges(torch.Tensor(mesh.cells[1-k].data)).long()
    else:
        raise ValueError("The dimension must be either 2 or 3.")
    # get edge attributes
    u_i = mesh.points[edge_index[0]][:,:config['graphnet']['dim']]
    u_j = mesh.points[edge_index[1]][:,:config['graphnet']['dim']]
    u_ij = torch.Tensor(u_i - u_j)
    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
    edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

    return Data(
        x=x.to(config['device']),
        edge_index=edge_index.to(config['device']),
        edge_attr=edge_attr.to(config['device']),
        cells=torch.Tensor(mesh.cells[1-k].data).to(config['device']),
        mesh_pos=torch.Tensor(mesh.points).to(config['device']),
        v_0=v_0.to(config['device']),
        name=config['name']
    )

def vtu(
    config: dict
) -> Data:
    # read vtu file
    mesh = meshio.read(osp.join(config['predict_dir'], 'data', 'cad_{:03d}.vtu'.format(config["name"])))

    # node type
    node_type = torch.zeros(mesh.points.shape[0])
    for i in range(mesh.cells[1].data.shape[0]):
        for j in range(mesh.cells[1].data.shape[1]):
            if (config['graphnet']['dim'] == 2):
                node_type[mesh.cells[1].data[i,j]] = mesh.cell_data['Label'][1][i]
            elif (config['graphnet']['dim']==3):
                if ((mesh.cell_data['Label'][1][i]==31) or (mesh.cell_data['Label'][1][i]==32)):
                    node_type[mesh.cells[1].data[i,j]] = 3
                else:
                    node_type[mesh.cells[1].data[i,j]] = mesh.cell_data['Label'][1][i] - 1
            else:
                raise ValueError("The dimension must be either 2 or 3.")

    # get initial velocity
    v_0 = torch.zeros(mesh.points.shape[0], config['graphnet']['dim'])
    mask = (node_type.long())==torch.tensor(NodeType.INFLOW)
    if (config['graphnet']['dim'] == 2):
        v_0[mask] = torch.Tensor([config['graphnet']['u_0'], config['graphnet']['v_0']])
    elif (config['graphnet']['dim'] == 3):
        v_0[mask] = torch.Tensor([config['graphnet']['u_0'], config['graphnet']['v_0'], config['graphnet']['w_0']])
    else:
        raise ValueError("The dimension must be either 2 or 3.")

    node_type_one_hot = torch.nn.functional.one_hot(node_type.long(), num_classes=NodeType.SIZE)

    # get features
    x = torch.cat((v_0, node_type_one_hot),dim=-1).type(torch.float)

    # get edge indices in COO format
    if (config['graphnet']['dim'] == 2):
        edge_index = triangles_to_edges(torch.Tensor(mesh.cells[0].data)).long()
    elif (config['graphnet']['dim'] == 3):
        edge_index = tetra_to_edges(torch.Tensor(mesh.cells[0].data)).long()
    else:
        raise ValueError("The dimension must be either 2 or 3.")

    # get edge attributes
    u_i = mesh.points[edge_index[0]][:,:config['graphnet']['dim']]
    u_j = mesh.points[edge_index[1]][:,:config['graphnet']['dim']]
    u_ij = torch.Tensor(u_i - u_j)
    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
    edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

    return Data(
        x=x.to(config['device']),
        edge_index=edge_index.to(config['device']),
        edge_attr=edge_attr.to(config['device']),
        cells=torch.Tensor(mesh.cells[0].data).to(config['device']),
        mesh_pos=torch.Tensor(mesh.points).to(config['device']),
        v_0=v_0.to(config['device']),
        name=config['name']
    )