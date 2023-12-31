import yaml
import os
import os.path as osp
import sys
import torch
import meshio
import time

from utils.utils import write_metric
import utils.process.meshnet as meshnet_process
import utils.process.graphnet as graphnet_process

if __name__ == '__main__':
    print('*** ADAPTNET ***\n')

    # load config file
    with open('src/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded config file from src/configs/config.yaml')
    # add path to python path
    sys.path.append(config['wdir'])

    os.makedirs(osp.join(config['save_dir'], config['save_folder']), exist_ok=True)

    # import modules
    from meshnet.model.module import MeshNet
    from meshnet.utils.utils import generate_mesh_2d, generate_mesh_3d
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

    # time the execution
    start_time = time.time()
    print('MeshNet...')
    # process cad
    processed_cad = meshnet_process.file(config=config)

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

    print(f'Done in: {time.time() - start_time:.2f}s\n')

    # save prediction to txt file
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'txt'), exist_ok=True)
    with open(osp.join(config['save_dir'], config['save_folder'], 'txt', 'cad_{:03d}.txt'.format(config["name"])), 'w') as f:
        for i in range(pred.shape[0]):
            f.write('{:.6f}\n'.format(pred[i][0]))

    # create mesh directories
    os.makedirs(osp.join(config['save_dir'], config['save_folder']), exist_ok=True)
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'vtk'), exist_ok=True)
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'mesh'), exist_ok=True)

    # save mesh
    if (config['meshnet']['dim'] == 2):
        generate_mesh_2d(
            cad_path=osp.join(config['predict_dir'], 'data', 'cad_{:03d}'.format(config['name']), 'cad_{:03d}.geo'.format(config['name'])),
            batch=processed_cad,
            pred=pred,
            save_dir=osp.join(config['save_dir'], config['save_folder'])
        )
    elif (config['meshnet']['dim'] == 3):
        generate_mesh_3d(
            cad_path=osp.join(config['predict_dir'], 'data', 'cad_{:03d}'.format(config['name']), 'cad_{:03d}.geo'.format(config['name'])),
            batch=processed_cad,
            pred=pred,
            save_dir=osp.join(config['save_dir'], config['save_folder'])
        )
    else:
        raise ValueError("The dimension must be either 2 or 3.")

    start_time = time.time()
    print('GraphNet...')
    # read mesh
    processed_mesh = graphnet_process.vtk(config=config)

    # load stats
    train_stats, val_stats, test_stats = graphnet_stats.load_stats(config['graphnet']['data_dir'], torch.device(config['device']))
    mean_vec_x_train, std_vec_x_train, mean_vec_edge_train, std_vec_edge_train, mean_vec_y_train, std_vec_y_train = train_stats

    # predict velocity
    pred = graphnet_stats.unnormalize(
            data=graphnet(
                batch=processed_mesh,
                split='predict',
                mean_vec_x_predict=mean_vec_x_train,
                std_vec_x_predict=std_vec_x_train,
                mean_vec_edge_predict=mean_vec_edge_train,
                std_vec_edge_predict=std_vec_edge_train),
            mean=mean_vec_y_train,
            std=std_vec_y_train
        )
    
    # save field
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'field'), exist_ok=True)
    write_metric(osp.join(config['save_dir'], config['save_folder'], 'field'), pred.squeeze().detach().cpu().numpy(), 'cad_{:03d}'.format(config["name"]))

    print(f'Done in: {time.time() - start_time:.2f}s\n')
    
    # save solution
    os.makedirs(osp.join(config['save_dir'], config['save_folder'], 'vtu'), exist_ok=True)

    point_data={
        'm_pred': pred.detach().cpu().numpy()
        }

    if (config['meshnet']['dim']==2):
        mesh = meshio.Mesh(
                points=processed_mesh.mesh_pos.cpu().numpy(),
                cells={"triangle": processed_mesh.cells.cpu().numpy()},
                point_data=point_data
            )
    elif (config['meshnet']['dim']==3):
        mesh = meshio.Mesh(
                points=processed_mesh.mesh_pos.cpu().numpy(),
                cells={"tetra": processed_mesh.cells.long().cpu().numpy()},
                point_data=point_data
            )
    else:
        raise ValueError("The dimension must be either 2 or 3.")
        
    mesh.write(osp.join(config['save_dir'], config['save_folder'], 'vtu', 'cad_{:03d}.vtu'.format(config["name"])), binary=False)

    print(f'Predictions saved in {osp.join(config["save_dir"], config["save_folder"])}')