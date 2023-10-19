import os.path as osp
import torch

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

def write_metric(path:str, m: torch.Tensor, name: str) -> None:
    with open(osp.join(path, f'{name}.sol'), 'w') as f:
        f.write('MeshVersionFormatted 1\n\n')
        f.write('Dimension 3\n\n')
        f.write('SolAtVertices\n')
        f.write(f'{len(m)}\n')
        f.write('1 1\n')
        for i in range(len(m)):
            f.write(f'{m[i]}\n')