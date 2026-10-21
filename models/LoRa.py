import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import argparse
import types
from models.vit import vit_base_patch16_dec512d8b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoRALayer(nn.Module):
    """
    LoRA implementation for linear layers
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        r: int = 8,  # Rank of LoRA decomposition
        lora_alpha: int = 32,  # Scaling factor
        lora_dropout: float = 0.1,  # Dropout probability for LoRA layers
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        
        # LoRA weight matrices
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # Optional dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward_ori(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout
        x_dropped = self.lora_dropout(x)
                
        # Ensure tensors are on the same device
        if x_dropped.device != self.lora_A.device:
            self.to(x_dropped.device)

        #debug
        print(f"LoRALayer: x_dropped shape: {x_dropped.shape}")
        print(f"LoRALayer: lora_A shape: {self.lora_A.shape}, lora_B shape: {self.lora_B.shape}")

        # Low-rank update: B·(A·x)
        return (self.lora_B @ (self.lora_A @ x_dropped.T)).T * self.scaling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout
        x_dropped = self.lora_dropout(x)
                
        # Ensure tensors are on the same device
        if x_dropped.device != self.lora_A.device:
            self.to(x_dropped.device)
        
        # Save original shape info
        original_shape = x_dropped.shape
        
        # Reshape input to 2D: [batch_size*seq_len, hidden_dim]
        x_2d = x_dropped.reshape(-1, x_dropped.shape[-1])
        
        # Perform LoRA computation on 2D tensor
        # Compute (x_2d @ lora_A.T) @ lora_B.T which is equivalent to (lora_B @ lora_A @ x_2d.T).T
        lora_output = (x_2d @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        
        # Reshape back to original dimensions
        return lora_output.reshape(original_shape)

class LoRAAttention(nn.Module):
    """
    Attention with LoRA applied to query and value projections
    """
    def __init__(self, base_attention, rank=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.base_attention = base_attention
        
        # Extract dimensions from base attention
        hidden_size = base_attention.qkv.in_features
        head_size = base_attention.head_dim
        num_heads = base_attention.num_heads
        
        # Create LoRA layers for query and value projections
        self.q_lora = LoRALayer(
            hidden_size, head_size * num_heads, 
            r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        self.v_lora = LoRALayer(
            hidden_size, head_size * num_heads, 
            r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
    
    def forward(self, x):
        # Get original attention output
        base_output = self.base_attention(x)
        
        # Compute LoRA contributions
        # Note: This simplified implementation assumes you can modify the internal
        # attention mechanism. For timm's implementation, you may need to customize further.
        
        # Return combined output
        return base_output
    
def apply_lora_to_vit(
    model,
    rank=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["qkv"],  # Target specific modules like "qkv" weights
    freeze_original=True,  # Freeze original parameters
):
    """
    Apply LoRA to a Vision Transformer model
    """
    # Keep track of LoRA parameters
    lora_params = []
    
    # Process blocks
    for i, block in enumerate(model.blocks):
        # Apply LoRA to attention QKV projections
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            # Get dimensions
            in_dim = block.attn.qkv.in_features
            out_dim = block.attn.qkv.out_features
            
            # Create LoRA layers
            q_lora = LoRALayer(in_dim, out_dim // 3, r=rank, 
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            k_lora = LoRALayer(in_dim, out_dim // 3, r=rank, 
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            v_lora = LoRALayer(in_dim, out_dim // 3, r=rank, 
                            lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            
            # Store original forward method
            orig_forward = block.attn.qkv.forward
            
            # Add LoRA parameters to tracking list
            lora_params.extend(list(q_lora.parameters()))
            lora_params.extend(list(k_lora.parameters()))
            lora_params.extend(list(v_lora.parameters()))
            
            # Create new forward method with LoRA
            def new_forward(self, x):
                orig_output = orig_forward(x)
                q_output = q_lora(x)
                k_output = k_lora(x)
                v_output = v_lora(x)
                
                # Split original output into q, k, v
                orig_q, orig_k, orig_v = orig_output.chunk(3, dim=-1)
                
                # Add LoRA contribution
                return torch.cat([orig_q + q_output, 
                                 orig_k + k_output, 
                                 orig_v + v_output], dim=-1)
            
            # Replace forward method
            block.attn.qkv.forward = types.MethodType(new_forward, block.attn.qkv)
    
    # Freeze original parameters
    if freeze_original:
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable training only for LoRA parameters
        for param in lora_params:
            param.requires_grad = True
        
    
    return model, lora_params

def lora_finetune(args, model, data_loader, optimizer, device):
    """
    Fine-tune with LoRA
    """
    # Apply LoRA adaptation
    model, lora_params = apply_lora_to_vit(
        model, 
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Create optimizer that only updates LoRA parameters
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Rest of training logic
    # ...


    
    # Train model
    # ...

def get_args_parser():
    parser = argparse.ArgumentParser("segmentation", add_help=False)
    # Existing arguments...
    
    # Add LoRA parameters
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_rank", default=8, type=int, help="Rank for LoRA decomposition")
    parser.add_argument("--lora_alpha", default=16, type=int, help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="Dropout probability for LoRA")
    
    return parser