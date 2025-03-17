import torch

def get_patch_positions(img_size, patch_size, device='cuda'):
    """
    Return a tensor of shape (num_patches, 2) with (row, col) indices
    for each patch in the patch grid.
    """
    H = img_size // patch_size
    W = img_size // patch_size
    positions = []
    for r in range(H):
        for c in range(W):
            positions.append([r, c])
    # positions is a list of length H*W, each [r, c]
    patch_positions = torch.tensor(positions).to(device=device)  # shape (H*W, 2)
    return patch_positions


def compute_shape_bias_penalty_batched(
    patch_positions: torch.Tensor,   # (B, N, 2)
    patch_embeddings: torch.Tensor,  # (B, N, d)
    alpha: float = 0.5,
    dist_scale: float = 1.0
) -> torch.Tensor:
    """
    Computes a shape-bias penalty in batch. Returns shape (B, N, N).

    patch_positions[b, i] = (row_i, col_i) for the i-th patch of the b-th image.
    patch_embeddings[b, i] = d-dim embedding for the i-th patch of the b-th image.
    """

    device = patch_positions.device
    dtype = patch_embeddings.dtype

    # Convert scalars alpha, dist_scale into Tensors on the same device/dtype
    alpha_tensor = torch.tensor(alpha, device=device, dtype=dtype)
    dist_scale_tensor = torch.tensor(dist_scale, device=device, dtype=dtype)

    # ---------------
    # (1) Correlation
    # ---------------
    # patch_embeddings: (B, N, d)
    # Normalize each patch embedding in dimension d to get cos similarity
    norms = patch_embeddings.norm(dim=2, keepdim=True) + 1e-6  # (B, N, 1)
    normalized = patch_embeddings / norms                      # (B, N, d)

    # We want pairwise dot products => shape (B, N, N)
    # Using einsum: (B,N,d) x (B,N,d) -> (B,N,N)
    corr_matrix = torch.einsum('b i d, b j d -> b i j', normalized, normalized)

    # -------------------
    # (2) Pairwise Distances
    # -------------------
    # patch_positions: (B, N, 2)
    # For row coords: shape (B, N), do unsqueeze to get (B, N, 1) - (B, 1, N) => (B, N, N)
    row_coords = patch_positions[..., 0]  # (B, N)
    col_coords = patch_positions[..., 1]  # (B, N)

    row_diff = row_coords.unsqueeze(2) - row_coords.unsqueeze(1)  # (B, N, N)
    col_diff = col_coords.unsqueeze(2) - col_coords.unsqueeze(1)  # (B, N, N)
    dist_matrix = torch.sqrt(row_diff**2 + col_diff**2)           # (B, N, N)

    # distance-based weight
    dist_weight = 1.0 / (1.0 + dist_scale_tensor * dist_matrix)   # (B, N, N)

    # ---------------
    # (3) Combine
    # ---------------
    penalty = alpha_tensor * corr_matrix * dist_weight  # (B, N, N)

    # ---------------
    # (4) Zero out diagonal
    # ---------------
    # We need to zero each batch's (N, N) diagonal
    B, N, _ = penalty.shape
    diag_idx = torch.arange(N, device=device)
    # We'll do this in a loop or fancy indexing.
    # "penalty[b, diag, diag] = 0" for each b in [0..B-1]
    penalty[:, diag_idx, diag_idx] = 0.0
    #print(penalty_batched.median(), penalty_batched.max(), penalty_batched.min())
    return penalty  # (B, N, N)
