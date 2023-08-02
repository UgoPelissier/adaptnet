import yaml
import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
import shutil
import torch
from torch_geometric.data import Data
import meshio
import time
from pyfreefem import FreeFemRunner

from utils.utils import length, triangles_to_edges, write_field

if __name__ == '__main__':
    print('*** ADAPTNET ***\n')
    # time the execution
    start_time = time.time()

    # load config file
    with open('predict/configs/mines.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded config file from predict/configs/mines.yaml')
    # add path to python path
    sys.path.append(config['wdir'])

    # import modules
    from meshnet.model.module import MeshNet
    from graphnet.model.module import GraphNet
    from graphnet.data.dataset import NodeType
    from graphnet.utils.stats import load_stats, unnormalize

    # load pre-trained models
    meshnet = MeshNet.load_from_checkpoint(
        checkpoint_path=config['meshnet']['checkpoint_path'],
        wdir=config['meshnet']['wdir'],
        data_dir=config['meshnet']['data_dir'],
        logs=config['meshnet']['logs'],
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
        num_layers=config['graphnet']['num_layers'],
        input_dim_node=config['graphnet']['input_dim_node'],
        input_dim_edge=config['graphnet']['input_dim_edge'],
        hidden_dim=config['graphnet']['hidden_dim'],
        output_dim=config['graphnet']['output_dim'],
        optimizer=config['graphnet']['optimizer'],
    )
    print(f'Loaded GraphNet from {config["graphnet"]["checkpoint_path"]}\n')

    print('MeshNet...')
    # load cad file
    df = pd.read_csv(osp.join(config['save_dir'], 'data', f'{config["name"]}.txt'), sep='\t')

    # process cad file as in meshnet dataset
    df['length'] = length(df)
    df['orientation'] = df['sens']
    
    x = torch.tensor(np.array(df.drop(columns=['xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend', 'label', 'sens'])), dtype=torch.float32).to(config['device'])
    pos = torch.tensor(df[['xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend']].values).to(config['device'])

    processed_cad = Data(x=x, pos=pos, name=torch.tensor(int(config['name'][-3:]), dtype=torch.long))

    # predict mesh point density
    pred = meshnet(processed_cad)
    data = torch.hstack((x, pos, pred))

    columns=['type', 'tstart', 'tend', 'radius1', 'radius2', 'length', 'orientation', 'xstart', 'ystart', 'zstart', 'xend', 'yend', 'zend', 'pred']
    type = data[:,0].long()
    tstart = data[:,1]
    tend = data[:,2]
    radius1 = data[:,3]
    radius2 = data[:,4]
    length = data[:,5]
    orientation = data[:,6]
    xstart = data[:,7]
    ystart = data[:,8]
    zstart = data[:,9]
    xend = data[:,10]
    yend = data[:,11]
    zend = data[:,12]
    pred = data[:,13]
    
    points = orientation*length/pred
    label = torch.Tensor(df['label'].values).long().to(config['device'])
    
    # generate mesh
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'msh'), exist_ok=True)
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'vtu'), exist_ok=True)
    runner = FreeFemRunner(script=osp.join(config['predict_dir'], 'freefem', 'prim2mesh.edp'), run_dir=osp.join(config['save_dir'], 'tmp', 'meshnet'))
    runner.import_variables(
        path=osp.join(config['save_dir'], config['save_folder']),
        name=config['name'],
        type=type,
        tstart=tstart,
        tend=tend,
        xs=xstart,
        ys=ystart,
        zs=zstart,
        xe=xend,
        ye=yend,
        ze=zend,
        r1=radius1,
        r2=radius2,
        label=label,
        points=points.long()
        )
    runner.execute()

    # clean tmp folder
    shutil.rmtree(osp.join(config['save_dir'], 'tmp', 'meshnet'))
    print('Done\n')

    print('GraphNet...')
    # read mesh
    mesh = meshio.read(osp.join(config['save_dir'], config['save_folder'], 'vtu', f'{config["name"]}.vtu'))

    # node type
    node_type = torch.zeros(mesh.points.shape[0])
    for i in range(mesh.cells[1].data.shape[0]):
        for j in range(mesh.cells[1].data.shape[1]):
            tmp = mesh.cell_data['Label'][1][i]
            if (tmp<4):
                node_type[mesh.cells[1].data[i,j]] = tmp
            else:
                node_type[mesh.cells[1].data[i,j]] = tmp - 1
            
    node_type_one_hot = torch.nn.functional.one_hot(node_type.long(), num_classes=NodeType.SIZE)

    # get initial velocity
    v_0 = torch.zeros(mesh.points.shape[0], 2)
    mask = (node_type.long())==torch.tensor(NodeType.INFLOW)
    v_0[mask] = torch.Tensor([1.0, 0.0])

    # get features
    x = torch.cat((v_0, node_type_one_hot),dim=-1).type(torch.float)

    # get edge indices in COO format
    edge_index =triangles_to_edges(torch.Tensor(mesh.cells[0].data)).long()

    # get edge attributes
    u_i = mesh.points[edge_index[0]][:,:2]
    u_j = mesh.points[edge_index[1]][:,:2]
    u_ij = torch.Tensor(u_i - u_j)
    u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
    edge_attr = torch.cat((u_ij, u_ij_norm),dim=-1).type(torch.float)

    mesh_processed = Data(
        x=x.to(config['device']),
        edge_index=edge_index.to(config['device']),
        edge_attr=edge_attr.to(config['device']),
        cells=torch.Tensor(mesh.cells[0].data).to(config['device']),
        mesh_pos=torch.Tensor(mesh.points).to(config['device']),
        v_0=v_0.to(config['device']),
        name=data[:-4]
    )

    # normalize node features
    mean_vec_x = torch.sum(x, dim = 0)/x.shape[0]
    std_vec_x = torch.maximum(torch.sqrt(torch.sum(x**2, dim = 0) / x.shape[0] - mean_vec_x**2), torch.tensor(1e-8))

    # normalize edge features
    mean_vec_edge = torch.sum(edge_attr, dim = 0)/edge_attr.shape[0]
    std_vec_edge = torch.maximum(torch.sqrt(torch.sum(edge_attr**2, dim = 0) / edge_attr.shape[0] - mean_vec_edge**2), torch.tensor(1e-8))

    # load stats
    train_stats, val_stats, test_stats = load_stats(config['graphnet']['data_dir'], torch.device(config['device']))
    mean_vec_x_train, std_vec_x_train, mean_vec_edge_train, std_vec_edge_train, mean_vec_y_train, std_vec_y_train = train_stats

    # predict velocity
    pred = unnormalize(
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
    mesh = meshio.Mesh(
            points=mesh_processed.mesh_pos.cpu().numpy(),
            cells={"triangle": mesh_processed.cells.cpu().numpy()},
            point_data={'u_pred': pred[:,0].detach().cpu().numpy(),
                        'v_pred': pred[:,1].detach().cpu().numpy()}
        )
    mesh.write(osp.join(config['save_dir'], config['save_folder'], 'vtu', f'{config["name"]}_sol.vtu'), binary=False)

    # save field
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'field'), exist_ok=True)
    write_field(osp.join(config['save_dir'], config['save_folder'], 'field'), pred[:,0], 'u_pred')
    write_field(osp.join(config['save_dir'], config['save_folder'], 'field'), pred[:,1], 'v_pred')

    # adapt mesh
    runner = FreeFemRunner(script=osp.join(config['predict_dir'], 'freefem', 'adapt.edp'), run_dir=osp.join(config['save_dir'], 'tmp', 'graphnet'))
    runner.import_variables(
            mesh_dir=osp.join(config['save_dir'], config['save_folder'], 'msh'),
            name=config['name'],
            wdir=osp.join(config['save_dir'], config['save_folder']),
            field_dir=osp.join(config['save_dir'], config['save_folder'], 'field'),
            )
    runner.execute()

    # clean tmp folder
    shutil.rmtree(osp.join(config['save_dir'], 'tmp', 'graphnet'))
    print('Done\n')

    print(f'Predictions saved in {osp.join(config["save_dir"], config["save_folder"])}')
    print(f'Execution time: {time.time() - start_time:.2f}s\n')