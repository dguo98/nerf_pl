import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils.ops import grid_sample_gradfix

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def generate_planes_sphere(h_stddev, v_stddev, num=12, r=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5):
    h = (torch.rand((num, 1)) - .5) * 2 * h_stddev + h_mean
    v = (torch.rand((num, 1)) - .5) * 2 * v_stddev + v_mean
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    theta = h
    v = v / math.pi
    phi = torch.arccos(1 - 2 * v)

    # Compute plane translation and plane normal.
    plane_origins = torch.zeros((num, 3))
    plane_origins[:, 0:1] = r * torch.sin(phi) * torch.cos(theta)
    plane_origins[:, 2:3] = r * torch.sin(phi) * torch.sin(theta)
    plane_origins[:, 1:2] = r * torch.cos(phi)

    # Normalize plane normal
    plane_normals = normalize_vecs(-plane_origins)

    # Compute plane axes.
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float).expand_as(plane_normals)

    left_vector = normalize_vecs(torch.cross(up_vector, plane_normals, dim=-1))
    up_vector = normalize_vecs(torch.cross(plane_normals, left_vector, dim=-1))

    rotation_matrix = torch.eye(4).unsqueeze(0).repeat(num, 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -plane_normals), axis=-1)

    translation_matrix = torch.eye(4).unsqueeze(0).repeat(num, 1, 1)
    translation_matrix[:, :3, 3] = plane_origins

    # Num planes x 4 x 4
    cam2world = translation_matrix @ rotation_matrix
    return cam2world

def compute_cam2world(phi, theta, r, up_axis):
    plane_origins = torch.zeros((1, 3))
    plane_origins[:, 0:1] = r * torch.sin(phi) * torch.cos(theta)
    plane_origins[:, 2:3] = r * torch.sin(phi) * torch.sin(theta)
    plane_origins[:, 1:2] = r * torch.cos(phi)

    # Normalize plane normal
    plane_normals = normalize_vecs(-plane_origins)

    # Compute plane axes.
    up_vector = torch.tensor(up_axis, dtype=torch.float).expand_as(plane_normals)

    left_vector = normalize_vecs(torch.cross(up_vector, plane_normals, dim=-1))
    up_vector = normalize_vecs(torch.cross(plane_normals, left_vector, dim=-1))

    rotation_matrix = torch.eye(4).unsqueeze(0)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -plane_normals), axis=-1)

    translation_matrix = torch.eye(4).unsqueeze(0)
    translation_matrix[:, :3, 3] = plane_origins

    # Num planes x 4 x 4
    cam2world = translation_matrix @ rotation_matrix
    return cam2world

def generate_planes_cube(r=0.914):
    planes = []

    # phi = 0, theta = 0 --> [0, 1, 0] --> XZ plane
    # Compute plane translation and plane normal.
    phi = torch.FloatTensor([0])
    theta = torch.FloatTensor([0])
    planes.append(compute_cam2world(phi, theta, r, [0, 0, 1]))
    planes.append(compute_cam2world(phi, theta, -r, [0, 0, 1]))

    # phi = 90, theta = 0 --> [1, 0, 0] --> YZ plane
    phi = torch.FloatTensor([math.pi / 2])
    theta = torch.FloatTensor([0])
    planes.append(compute_cam2world(phi, theta, r, [0, 1, 0]))
    planes.append(compute_cam2world(phi, theta, -r, [0, 1, 0]))

    # phi = 90, theta = 90 --> [0, 0, 1] --> XY plane
    phi = torch.FloatTensor([math.pi / 2])
    theta = torch.FloatTensor([math.pi / 2])
    planes.append(compute_cam2world(phi, theta, r, [0, 1, 0]))
    planes.append(compute_cam2world(phi, theta, -r, [0, 1, 0]))

    planes = torch.cat(planes, dim=0)
    return planes

def generate_planes(num=12):
    if num == 1:
        return torch.tensor([[[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]], dtype=torch.float32)
    elif num == 3:
        return torch.tensor([[[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             [[1, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]],
                             [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]]], dtype=torch.float32)
    else:
        unormalized = torch.randn(num, 3, 3)
        Q = torch.qr(unormalized).Q
        return Q

def project_onto_planes(planes, coordinates, perspective=False):
    # Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.inverse(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes.permute(0, 2, 1))

    if perspective:
        # projections is N*n_planes x M x 3
        ones = torch.ones(projections.shape[0], projections.shape[1], 1).to(coordinates.device)
        projections = torch.cat([projections, ones], dim=2)

        fov = 90
        near = 0
        far = 1
        S = 1.0 / math.tan((fov / 2) * (math.pi / 180))
        perspective_mat = torch.zeros(4, 4).to(projections.device)
        perspective_mat[0, 0] = S
        perspective_mat[1, 1] = S
        perspective_mat[2, 2] = -far / (far - near)
        perspective_mat[2, 3] = -1
        perspective_mat[3, 2] = -(far * near) / (far - near)

        # Apply perspective projection matrix and perspective divide.
        perspective_mat = perspective_mat.unsqueeze(0).expand(N*n_planes, -1, -1)
        perspective_projection = torch.bmm(projections, perspective_mat)
        perspective_projection = perspective_projection / perspective_projection[:, :, 3].unsqueeze(2)
        perspective_projection = perspective_projection[:, :, :3]

        # filter out possible divide by 0 when z-value is 0.
        perspective_projection[torch.isnan(perspective_projection)] = 0
        perspective_projection[torch.isinf(perspective_projection)] = 0

        projections = perspective_projection

    plane_distances = projections[..., 2]
    plane_distances = plane_distances.reshape(N, n_planes, M)

    return projections[..., :2], plane_distances


import torch
import torch.nn.functional as F

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def sample_from_axis(plane_axes, plane_features, coordinates, mode='bilinear', use_custom=False):
    N, n_planes, C, H = plane_features.shape
    _, M, _ = coordinates.shape
    assert N == 1, "sample from axis only supports batch size (N) = 1"

    plane_features = plane_features.view(N*n_planes, C, H, 1)

    projected_coordinates = coordinates.permute(2, 0, 1)
    assert projected_coordinates.shape == (n_planes, N, M)
    projected_coordinates = torch.stack([-torch.ones_like(projected_coordinates), projected_coordinates], dim=-1)
    assert projected_coordinates.shape == (n_planes, N, M, 2)

    #projected_coordinates, plane_distances = project_onto_planes(plane_axes, coordinates, perspective=False)
    #projected_coordinates = projected_coordinates.unsqueeze(1)
    #print("plane_features.shape=", plane_features.shape, " projected_coordinates", projected_coordinates.shape)
    #sys.exit(0)

    # output_features = F.grid_sample(plane_features.float(), projected_coordinates.float(), mode=mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    # output_features = grid_sample_gradfix.grid_sample(plane_features.float(), projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    if use_custom:
        # print('using custom')
        output_features = grid_sample(plane_features.float(), projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    else:
        output_features = F.grid_sample(plane_features.float(), projected_coordinates.float(), mode='bilinear', align_corners=True, padding_mode='border').permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features, None 


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', use_custom=False):
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    plane_features = plane_features.view(N*n_planes, C, H, W)
    projected_coordinates, plane_distances = project_onto_planes(plane_axes, coordinates, perspective=False)
    projected_coordinates = projected_coordinates.unsqueeze(1)

    # output_features = F.grid_sample(plane_features.float(), projected_coordinates.float(), mode=mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    # output_features = grid_sample_gradfix.grid_sample(plane_features.float(), projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    if use_custom:
        # print('using custom')
        output_features = grid_sample(plane_features.float(), projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    else:
        output_features = F.grid_sample(plane_features.float(), projected_coordinates.float(), mode='bilinear', align_corners=True, padding_mode='border').permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features, plane_distances

def project_onto_sphere_planes(planes, coordinates):
    # Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    ones = torch.ones(N*n_planes, M, 1).to(coordinates.device)
    coordinates = torch.cat([coordinates, ones], dim=2)
    inv_planes = torch.inverse(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 4, 4)
    projections = torch.bmm(coordinates, inv_planes.permute(0, 2, 1))

    plane_distances = projections[..., 2]
    plane_distances = plane_distances.reshape(N, n_planes, M)
    return projections[..., :2], plane_distances

def sample_from_sphere_planes(plane_axes, plane_features, coordinates, mode='bilinear'):
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    plane_features = plane_features.view(N*n_planes, C, H, W)
    #projected_coordinates, plane_distances = project_onto_sphere_planes(plane_axes, coordinates)
    projected_coordinates, plane_distances = perspective_project_onto_planes(plane_axes, coordinates)
    projected_coordinates = projected_coordinates.unsqueeze(1)

    output_features = F.grid_sample(plane_features.float(), projected_coordinates.float(), mode=mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features, plane_distances

def perspective_project_onto_planes(planes, coordinates):
    # Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    ones = torch.ones(N*n_planes, M, 1).to(coordinates.device)
    coordinates = torch.cat([coordinates, ones], dim=2)
    inv_planes = torch.inverse(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 4, 4)
    projections = torch.bmm(coordinates, inv_planes.permute(0, 2, 1))

    # projections is N*n_planes x M x 4
    fov = 90
    near = 0.6
    far = 1.8
    S = 1.0 / math.tan((fov / 2) * (math.pi / 180))
    perspective_mat = torch.zeros(4, 4).to(projections.device)
    perspective_mat[0, 0] = S
    perspective_mat[1, 1] = S
    perspective_mat[2, 2] = -far / (far - near)
    perspective_mat[2, 3] = -1
    perspective_mat[3, 2] = -(far * near) / (far - near)

    perspective_mat = perspective_mat.unsqueeze(0).expand(N*n_planes, -1, -1)
    perspective_projection = torch.bmm(projections, perspective_mat)
    perspective_projection = perspective_projection / perspective_projection[:, :, 3].unsqueeze(2)
    perspective_projection = perspective_projection[:, :, :3]

    # multiply by 1 x 4 x 4 perspective projection matrix
    return perspective_projection[..., :2], perspective_projection[..., 2].reshape(N, n_planes, M)

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class NeRF_Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_nerf_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = NeRF_Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class FourierFeatureTransformAggregation(nn.Module):
    def __init__(self, num_planes, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_planes, num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x, weights):
        B, N, n_planes, C = x.shape

        # (n_planes x B*N x C) * (n_planes x C x mapping_size)
        x = x.reshape(B*N, n_planes, C).permute(1, 0, 2)
        x = torch.matmul(x, self._B)
        x = x.permute(1, 0, 2).reshape(B, N, n_planes, -1)

        # Apply each plane's weights.
        x = weights.unsqueeze(3) * x
        x = torch.sum(x, dim=2)

        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var

class IBRAggregator(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()
        activation_func = nn.ELU(inplace=True)
        self.base_fc = nn.Sequential(nn.Linear(num_input_channels*3, 64),
                                     activation_func,
                                     nn.Linear(64, num_output_channels + 1),
                                     activation_func)
        self.geometry_fc = nn.Sequential(nn.Linear(2*num_output_channels+1, 64),
                                         activation_func,
                                         nn.Linear(64, num_output_channels),
                                         activation_func)

        # Initialize weights.
        self.base_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)

    def forward(self, x):
        num_views = x.shape[2]

        # Compute mean and variance of features.
        weights = torch.ones_like(x)
        mean, var = fused_mean_variance(x, weights)                           # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)                           # [n_rays, n_samples, 1, 2*n_feat]

        # Compute multi-view aware features and weights.
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), x], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)                                                   # [n_rays, n_samples, n_views, output_feat+1]

        # Split into features and weights.
        x, weights = torch.split(x, [x.shape[-1]-1, 1], dim=-1)
        weights = torch.sigmoid(weights)

        # Use weights to compute weighted mean and variance of multi-view aware features.
        mean, var = fused_mean_variance(x, weights)

        # Predict final output feature using weighted mean, weighted variance, and weights.
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weights.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 2*output_feat+1]
        globalfeat = self.geometry_fc(globalfeat)                                               # [n_rays, n_samples, output_feat]

        return globalfeat
