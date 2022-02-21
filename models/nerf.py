import torch
from torch import nn
from models.planes import sample_from_planes
from models.planes import generate_planes
from models.planes import FourierFeatureTransform 

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        #print("!!sidelength=", sidelength, " scale_factor=", self.scale_factor)

    def forward(self, coordinates):
        return coordinates * self.scale_factor


class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation="LeakyReLU"):
        super().__init__()
        self.net = nn.Linear(input_dim, hidden_dim)
        if activation=="Softplus":
            self.activation = nn.Softplus(beta=100)
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.hidden_dim = hidden_dim

    def forward(self, x, mod):
        gamma, beta = mod[..., :self.hidden_dim].unsqueeze(-2), mod[..., self.hidden_dim:].unsqueeze(-2)
        x = self.net(x)
        x = x * (1+gamma) + beta
        return self.activation(x)

class NeRFCube(nn.Module):
    def __init__(self, hparams, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.hparams = hparams  # used for additiona params
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = 3  # original coordinates
        self.in_channels_dir = in_channels_dir
        self.mode = hparams.mode

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # xyz encoding layers
        self.n_planes = 3
        self.n_features = self.hparams.n_features
        self.n_xyz_dim = self.hparams.n_xyz_dim  # hidden layers for plane features
        self.use_xyz_net = self.hparams.use_xyz_net
        if self.use_xyz_net == 0:
            self.n_xyz_dim = W  # override
        self.backbone_resolution = self.hparams.backbone_res
        self.box_warp = self.hparams.box_warp # todo(demi): figure out box_warp
        
        self.cube_features = nn.Parameter(torch.randn(1, self.n_features, self.backbone_resolution, self.backbone_resolution, self.backbone_resolution)*0.01)

        self.gridwarper = UniformBoxWarp(self.box_warp)
        self.pe = FourierFeatureTransform(self.n_features+3, self.n_xyz_dim // 2)  # (concat) agged features + original coordinates 
        
        # NB(demi): optional, to match triplane
        if self.mode == "debug":
            mid_channels_xyz = self.n_xyz_dim
            for i in range(D):
                if i == 0:
                    layer = nn.Linear(mid_channels_xyz, W)
                elif i in skips:
                    layer = nn.Linear(W+mid_channels_xyz, W)
                else:
                    layer = nn.Linear(W, W)
                layer = nn.Sequential(layer, nn.ReLU(True))
                setattr(self, f"xyz_encoding_{i+1}", layer)
        else:
            assert self.mode == "default"
            if self.use_xyz_net == 1:
                self.xyz_net = nn.Sequential(nn.Linear(self.n_xyz_dim, self.n_xyz_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.n_xyz_dim, W),
                nn.LeakyReLU(0.2, inplace=True)
                )
            
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
            

        xyz_ = input_xyz

        coordinates = self.gridwarper(xyz_)
        N = 1
        M = coordinates.shape[0]
        C = self.n_features
        assert coordinates.shape == (M,3)
        
        coordinates = coordinates.reshape(1, M, 3)
        grid = coordinates.reshape((N,1,1,M,3))
        sampled_features = nn.functional.grid_sample(self.cube_features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_features = sampled_features.permute((0, 4, 1, 2, 3))
        sampled_features = sampled_features.reshape(N, M, -1)
       
        sampled_features = torch.cat([sampled_features, coordinates], -1)
        assert sampled_features.shape == (N, M, C+3)

        xyz_ = self.pe(sampled_features)
        assert xyz_.shape == (1, M, self.n_xyz_dim)  # final xyz features

        if self.mode == "debug":
            xyz_ = xyz_.reshape(M, self.n_xyz_dim)
            input_xyz = xyz_    
            for i in range(self.D):
                if i in self.skips:
                    xyz_ = torch.cat([input_xyz, xyz_], 1)
                xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        else:
            if self.use_xyz_net == 1:
                xyz_ = self.xyz_net(xyz_)
                assert xyz_.shape == (1, M, self.W)
                
            xyz_ = xyz_.reshape(M, self.W)
            
        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)

class NeRFTriplane(nn.Module):
    def __init__(self, hparams, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.hparams = hparams  # used for additiona params
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = 3  # original coordinates
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # xyz encoding layers
        self.n_planes = 3
        self.n_features = self.hparams.n_features
        self.n_xyz_dim = self.hparams.n_xyz_dim  # hidden layers for plane features
        self.use_xyz_net = self.hparams.use_xyz_net
        if self.use_xyz_net == 0:
            self.n_xyz_dim = W  # override
        self.backbone_resolution = self.hparams.backbone_res
        self.box_warp = self.hparams.box_warp # todo(demi): figure out box_warp
        self.plane_features = nn.Parameter(torch.randn(1,self.n_planes, self.n_features, self.backbone_resolution, self.backbone_resolution)*0.01)
        self.plane_axes = nn.Parameter(generate_planes(self.n_planes), requires_grad=False)
        self.gridwarper = UniformBoxWarp(self.box_warp)
        self.pe = FourierFeatureTransform(3*self.n_features+3, self.n_xyz_dim // 2)  # (concat) agged features + original coordinates 
        
        # NB(demi): optional, to match triplane
        if self.hparams.mode in ["debug5", "debug6"]:
            self.embedding_xyz = PosEmbedding(self.hparams.N_emb_xyz-1,self.hparams.N_emb_xyz)
            if self.use_xyz_net == 1:
                if self.hparams.mode == "debug5":
                    self.xyz_net = nn.Sequential(nn.Linear(self.n_xyz_dim+self.hparams.N_emb_xyz*6+3, self.n_xyz_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.n_xyz_dim, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
                else:
                    self.xyz_net = nn.Sequential(nn.Linear(self.n_xyz_dim+self.hparams.N_emb_xyz*6+3, self.n_xyz_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.n_xyz_dim, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    nn.Linear(W, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    nn.Linear(W, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    nn.Linear(W, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    nn.Linear(W, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    nn.Linear(W, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    nn.Linear(W, W),
                    nn.LeakyReLU(0.2, inplace=True)
                    )
        else:   
            if self.use_xyz_net == 1:
                self.xyz_net = nn.Sequential(nn.Linear(self.n_xyz_dim, self.n_xyz_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.n_xyz_dim, W),
                nn.LeakyReLU(0.2, inplace=True)
                )
            
        if self.hparams.mode in ["debug", "debug2"]:
            self.embedding_xyz = PosEmbedding(self.hparams.N_emb_xyz-1,self.hparams.N_emb_xyz)
            assert in_channels_xyz == 63  # still default, after pos embedding
            for i in range(D):
                if i == 0:
                    layer = nn.Linear(in_channels_xyz, W)
                elif i in skips:
                    layer = nn.Linear(W+in_channels_xyz, W)
                else:
                    layer = nn.Linear(W, W)
                layer = nn.Sequential(layer, nn.ReLU(True))
                setattr(self, f"xyz_encoding_{i+1}", layer)
            
            self.agg_layer = nn.Linear(2*W, W)
        elif self.hparams.mode in ["debug3", "debug4"]:
            in_channels_xyz=3
            for i in range(D):
                if i == 0:
                    layer = nn.Linear(in_channels_xyz, W)
                elif i in skips:
                    layer = nn.Linear(W+in_channels_xyz, W)
                else:
                    layer = nn.Linear(W, W)
                layer = nn.Sequential(layer, nn.ReLU(True))
                setattr(self, f"xyz_encoding_{i+1}", layer)
            
            self.agg_layer = nn.Linear(2*W, W)

        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
            

        xyz_ = input_xyz
        if self.hparams.mode in ["debug5", "debug6"]:
            old_xyz_ = self.embedding_xyz(input_xyz)

        # original NeRF
        if self.hparams.mode in ["debug", "debug2"]:
            input_xyz = self.embedding_xyz(input_xyz)
            old_xyz_ = input_xyz
            for i in range(self.D):
                if i in self.skips:
                    old_xyz_ = torch.cat([input_xyz, old_xyz_], 1)
                old_xyz_ = getattr(self, f"xyz_encoding_{i+1}")(old_xyz_)
            assert old_xyz_.shape[1] == self.W and len(old_xyz_.shape) == 2
        elif self.hparams.mode in ["debug3", "debug4"]:
            if self.hparams.mode == "debug3":
                coordinates = self.gridwarper(xyz_)
            else:
                assert self.hparams.mode == "debug4" 
                coordinates = xyz_
            input_xyz = coordinates
            old_xyz_ = input_xyz
            for i in range(self.D):
                if i in self.skips:
                    old_xyz_ = torch.cat([input_xyz, old_xyz_], 1)
                old_xyz_ = getattr(self, f"xyz_encoding_{i+1}")(old_xyz_)
            assert old_xyz_.shape[1] == self.W and len(old_xyz_.shape) == 2


        # triplane
        coordinates = self.gridwarper(xyz_)
        M = coordinates.shape[0]
        C = self.n_features
        assert coordinates.shape == (M,3)

        coordinates = coordinates.reshape(1, M, 3)
        sampled_features, plane_distances = sample_from_planes(self.plane_axes, self.plane_features, coordinates)
        assert sampled_features.shape == (1, 3, M, C)
        sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(1, M, 3, C)

        # NB(demi): use concat as aggregation for now
        sampled_features = sampled_features.reshape(1, M, 3*C)
        sampled_features = torch.cat([sampled_features, coordinates], -1)
        assert sampled_features.shape == (1, M, 3*C+3)
        xyz_ = self.pe(sampled_features)
        assert xyz_.shape == (1, M, self.n_xyz_dim)  # final xyz features
        
        if self.hparams.mode in ["debug5", "debug6"]:
            old_xyz_ = old_xyz_.reshape(1, M, -1)
            xyz_ = torch.cat([xyz_, old_xyz_], -1)

        if self.use_xyz_net == 1:
            xyz_ = self.xyz_net(xyz_)
            assert xyz_.shape == (1, M, self.W)
            
        xyz_ = xyz_.reshape(M, self.W)

        # DEBUG(demi): merge
        if self.hparams.mode == "debug":
            xyz_ = self.agg_layer(torch.cat([xyz_, old_xyz_],1))
            assert xyz_.shape == (M, self.W)
        elif self.hparams.mode == "debug2":
            xyz_ = old_xyz_
        
        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)

class NeRF(nn.Module):
    def __init__(self, hparams, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
            

        xyz_ = input_xyz
        #print("xyz_.shape=", xyz_.shape)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        
        #print("final xyz_ features before static sigma=", xyz_.shape)
        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)
