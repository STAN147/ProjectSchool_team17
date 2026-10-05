import torch
import torch.nn as nn

class ModernNCABlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ModernNCAEncoder(nn.Module):
    def __init__(
        self, 
        plr_module: nn.Module | None,  
        num_continuous: int,           
        plr_dim: int,                  
        num_categorical: int,          
        hidden_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        dropout_rate: float,
        post_mlp_layers: list[int] | None = None
    ):
        super().__init__()
        self.plr_module = plr_module

        in_dim = 0
        if num_continuous > 0:
            in_dim += num_continuous * (plr_dim if plr_module is not None else 1)
        if num_categorical > 0:
            in_dim += num_categorical
            
        if in_dim == 0:
            raise ValueError("Модель не может обучаться без признаков.")
        
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(ModernNCABlock(curr_dim, hidden_dim, embed_dim, dropout_rate))
            curr_dim = embed_dim
        
        self.network = nn.Sequential(*layers)

        mlp_blocks = []
        if post_mlp_layers is not None and len(post_mlp_layers) > 0:
            for mlp_out in post_mlp_layers:
                mlp_blocks.extend([
                    nn.Linear(curr_dim, mlp_out),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                curr_dim = mlp_out
                
        self.post_mlp = nn.Sequential(*mlp_blocks) if mlp_blocks else nn.Identity()

    def forward(
        self, 
        x_num: torch.Tensor | None, 
        x_cat: torch.Tensor | None
    ) -> torch.Tensor:
        features = []
    
        if x_num is not None:
            features.append(
                self.plr_module(x_num).flatten(start_dim=1)
                if self.plr_module is not None else x_num
            )
            
        if x_cat is not None:
            features.append(x_cat)
    
        x_concat = torch.cat(features, dim=1)
        encoded = self.network(x_concat)
        projected = self.post_mlp(encoded)
        return projected
