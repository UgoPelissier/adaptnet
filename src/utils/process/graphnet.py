import torch

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