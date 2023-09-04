import os.path as osp
import torch
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