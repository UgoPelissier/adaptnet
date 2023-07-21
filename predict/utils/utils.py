import os.path as osp
import numpy as np
import pandas as pd
import torch

def line(
        start: np.ndarray,
        end: np.ndarray,
        scale: float
) -> float:
    """Compute the length of a line."""
    return scale*float(np.linalg.norm(end-start))


def ellipse(
        r1: float,
        r2: float,
        scale: float
) -> float:
    """Compute the length of an ellipse."""
    return scale*np.sqrt((r1**2 + r2**2)/2)


def length(
        df: pd.DataFrame
) -> np.ndarray:
    """Compute the length of the primitive."""
    length = []
    for i in range(df.shape[0]):
        temp = df.iloc[i]
        if (temp['type'] == 1):
            length.append(line(np.array(temp[['xstart', 'ystart', 'zstart']].values), np.array(temp[['xend', 'yend', 'zend']].values), temp['tend']-temp['tstart']))
        elif (temp['type'] == 2):
            length.append(ellipse(temp['radius1'], temp['radius2'], temp['tend']-temp['tstart']))
    return np.array(length)


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