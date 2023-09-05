import yaml
import os
import os.path as osp
import sys
import torch
from torch_geometric.data import Data
import meshio
import time

from utils.utils import node_type, triangles_to_edges, write_field

if __name__ == '__main__':
    print('*** ADAPTNET ***\n')
    # time the execution
    start_time = time.time()

    # load config file
    with open('src/configs/mines.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded config file from src/configs/mines.yaml')
    # add path to python path
    sys.path.append(config['wdir'])

    os.makedirs(osp.join(config['save_dir'], config['save_folder']), exist_ok=True)

    # import modules
    from meshnet.model.module import MeshNet
    import meshnet.utils.stats as meshnet_stats
    from graphnet.model.module import GraphNet
    from graphnet.data.dataset import NodeType
    import graphnet.utils.stats as graphnet_stats

    # load pre-trained models
    meshnet = MeshNet.load_from_checkpoint(
        checkpoint_path=config['meshnet']['checkpoint_path'],
        wdir=config['meshnet']['wdir'],
        data_dir=config['meshnet']['data_dir'],
        logs=config['meshnet']['logs'],
        dim=config['meshnet']['dim'],
        num_layers= config['meshnet']['num_layers'],
        input_dim_node= config['meshnet']['input_dim_node'], 
        input_dim_edge= config['meshnet']['input_dim_edge'],
        hidden_dim= config['meshnet']['hidden_dim'],
        output_dim= config['meshnet']['output_dim'],
        optimizer=config['meshnet']['optimizer']
    )
    print(f'Loaded MeshNet from {config["meshnet"]["checkpoint_path"]}')

    graphnet = GraphNet.load_from_checkpoint(
        checkpoint_path=config['graphnet']['checkpoint_path'],
        dir=config['graphnet']['dir'],
        wdir=config['graphnet']['wdir'],
        data_dir=config['graphnet']['data_dir'],
        logs=config['graphnet']['logs'],
        dim=config['graphnet']['dim'],
        num_layers=config['graphnet']['num_layers'],
        hidden_dim=config['graphnet']['hidden_dim'],
        optimizer=config['graphnet']['optimizer'],
    )
    print(f'Loaded GraphNet from {config["graphnet"]["checkpoint_path"]}\n')

    print('MeshNet...')
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
        y = points[:, -1]
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

        processed_cad = Data(
            x=x.to(config['device']),
            edge_index=edge_index.to(config['device']),
            edge_attr=edge_attr.to(config['device']),
            y=y.to(config['device']),
            name=torch.Tensor([config['name']]).long().to(config['device'])
        )

    # load stats
    train_stats, val_stats, test_stats = meshnet_stats.load_stats(config['meshnet']['data_dir'], torch.device(config['device']))
    mean_vec_x_train, std_vec_x_train, mean_vec_edge_train, std_vec_edge_train, mean_vec_y_train, std_vec_y_train = train_stats

    # predict mesh point density    
    pred = meshnet_stats.unnormalize(
        data=meshnet(
            batch=processed_cad,
            split='predict',
            mean_vec_x_predict=mean_vec_x_train,
            std_vec_x_predict=std_vec_x_train,
            mean_vec_edge_predict=mean_vec_edge_train,
            std_vec_edge_predict=std_vec_edge_train
        ),
        mean=mean_vec_y_train,
        std=std_vec_y_train
    )

    # save mesh
    os.makedirs(osp.join(config['save_dir'], config['save_folder']), exist_ok=True)
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'vtk'), exist_ok=True)
    
    meshnet.generate_mesh(
        cad_path=osp.join(config['predict_dir'], 'data', 'cad_{:03d}.geo'.format(config['name'])),
        batch=processed_cad,
        pred=pred,
        save_dir=osp.join(config['save_dir'], config['save_folder'])
    )
    print('Done\n')

    print('GraphNet...')
    # read mesh
    mesh = meshio.read(osp.join(config['save_dir'], config['save_folder'], 'vtk', 'mesh_{:03d}.vtk'.format(config["name"])))

    # node type
    node_type = torch.zeros(mesh.points.shape[0])
    for i in range(mesh.cells[0].data.shape[0]):
        for j in range(mesh.cells[0].data.shape[1]):
            node_type[mesh.cells[0].data[i,j]] = int(mesh.cell_data['CellEntityIds'][1][i]) - 1
            
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
    v_0[mask] = torch.Tensor([1.0, 0.0])

    # get features
    x = torch.cat((v_0, node_type_one_hot),dim=-1).type(torch.float)

    # get edge indices in COO format
    edge_index = triangles_to_edges(dim=config['meshnet']['dim'], faces=torch.Tensor(mesh.cells[1].data)).long()

    # get edge attributes
    u_i = mesh.points[edge_index[0]][:,:config['graphnet']['dim']]
    u_j = mesh.points[edge_index[1]][:,:config['graphnet']['dim']]
    u_ij = torch.Tensor(u_i - u_j)
    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
    edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

    mesh_processed = Data(
        x=x.to(config['device']),
        edge_index=edge_index.to(config['device']),
        edge_attr=edge_attr.to(config['device']),
        cells=torch.Tensor(mesh.cells[1].data).to(config['device']),
        mesh_pos=torch.Tensor(mesh.points).to(config['device']),
        v_0=v_0.to(config['device']),
        name=config['name']
    )

    # load stats
    train_stats, val_stats, test_stats = graphnet_stats.load_stats(config['graphnet']['data_dir'], torch.device(config['device']))
    mean_vec_x_train, std_vec_x_train, mean_vec_edge_train, std_vec_edge_train, mean_vec_y_train, std_vec_y_train = train_stats

    # predict velocity
    pred = graphnet_stats.unnormalize(
            data=graphnet(
                batch=mesh_processed,
                split='predict',
                mean_vec_x_predict=mean_vec_x_train,
                std_vec_x_predict=std_vec_x_train,
                mean_vec_edge_predict=mean_vec_edge_train,
                std_vec_edge_predict=std_vec_edge_train),
            mean=mean_vec_y_train,
            std=std_vec_y_train
        )
    
    # save solution
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'vtu'), exist_ok=True)

    mesh = meshio.Mesh(
            points=mesh_processed.mesh_pos.cpu().numpy(),
            cells={"triangle": mesh_processed.cells.cpu().numpy()},
            point_data={'u_pred': pred[:,0].detach().cpu().numpy(),
                        'v_pred': pred[:,1].detach().cpu().numpy()}
        )
    mesh.write(osp.join(config['save_dir'], config['save_folder'], 'vtu', 'cad_{:03d}_sol.vtu'.format(config["name"])), binary=False)

    # save field
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'field'), exist_ok=True)
    write_field(osp.join(config['save_dir'], config['save_folder'], 'field'), pred[:,0], 'u_pred')
    write_field(osp.join(config['save_dir'], config['save_folder'], 'field'), pred[:,1], 'v_pred')

    print('Done\n')

    print(f'Predictions saved in {osp.join(config["save_dir"], config["save_folder"])}')
    print(f'Execution time: {time.time() - start_time:.2f}s\n')