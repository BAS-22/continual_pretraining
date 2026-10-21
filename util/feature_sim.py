import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.vit import vit_base_patch16_dec512d8b
from models.vit import ViT
import os
from models.LoRa import apply_lora_to_vit
from models.upernet_module import UperNet, UperNetSR, ViTAdapter
from models.build_segment import buidl_model
from monai.inferers import sliding_window_inference
import copy
from scipy.ndimage import zoom

def compute_kernel(X):
    """Compute the kernel matrix for CKA."""
    X = X.reshape(X.shape[0], -1)
    return X @ X.T

def centered_kernel_alignment(X, Y):
    """Compute CKA between two feature matrices."""
    K_X = compute_kernel(X)
    K_Y = compute_kernel(Y)
    
    # Center the kernel matrices
    n = K_X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H
    
    # Compute CKA
    cka = (np.linalg.norm(K_X_centered.flatten() @ K_Y_centered.flatten())) / \
          (np.linalg.norm(K_X_centered) * np.linalg.norm(K_Y_centered))
    return cka

def load_vit_models(args, SR=False):
    vit_pre = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
    vit_cont = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
    vit_pre_ft = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
    vit_cont_ft = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
    # if SR:
    #     vit_ft =ViT(
    #         img_size=(32, 256, 256),
    #         patch_size=(16, 32, 32),
    #         in_chans=1,
    #         embed_dim=768,
    #         depth=12,
    #         num_heads=12,
    #         mlp_ratio=4,
    #         norm_layer=nn.LayerNorm
    #     )
    # else:
    #     vit_ft = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
    # Apply LoRA
    vit_cont, lora_params = apply_lora_to_vit(
        vit_cont,            # The ViT model
        # rank=16,         # Rank of the decomposition
        rank=32,         # Rank of the decomposition
        lora_alpha=32,   # Scaling factor
        # lora_dropout=0.1, # Dropout probability
        lora_dropout=0.0, # Dropout probability
        freeze_original=False, # Freeze original parameters
    )

    vit_cont_ft, lora_params = apply_lora_to_vit(
        vit_cont_ft,            # The ViT model
        # rank=16,         # Rank of the decomposition
        rank=32,         # Rank of the decomposition
        lora_alpha=32,   # Scaling factor
        # lora_dropout=0.1, # Dropout probability
        lora_dropout=0.0, # Dropout probability
        freeze_original=False, # Freeze original parameters
    )

    return vit_pre, vit_cont, vit_pre_ft, vit_cont_ft

def resize_feature_map(feature_map, target_shape):
    """
    Resize a feature map to match target shape using interpolation.
    
    Args:
        feature_map: NumPy array of shape [C, H, W] or [H, W]
        target_shape: Target shape tuple
        
    Returns:
        Resized feature map with shape matching target_shape
    """
    
    # Get the zoom factors
    orig_shape = feature_map.shape
    zoom_factors = [t/o for t, o in zip(target_shape, orig_shape)]
    
    # Perform the resize using scipy's zoom
    resized_map = zoom(feature_map, zoom_factors, order=1)
    
    return resized_map

def compare_vit_encoders_with_cka_batch(args, model_paths, input_batch, sliding_window=False):
    """
    Compare three ViT encoders using CKA with a batch of inputs.
    """
    # Determine the device from input_batch
    device = input_batch.device
    print(f"Input batch is on device: {device}")
    
    vit_pre, vit_cont, vit_pre_ft, vit_cont_ft = load_vit_models(args)

    ####loading models
    # models = []
    # # define the model
    # device = torch.device("cuda")
    # model, optimizer = buidl_model(args=args, device=device) #pretrained weights loaded
    # model.eval()
    # filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    # # filepath_best = os.path.join("output_seg_lesion_pp_lora_dec799epoch/Promis/profound_vit/lora/2021", "best.pth.tar") # custom path
    # model.load_state_dict(torch.load(filepath_best)["model"])
    # print("loaded for test from ", filepath_best)
    # models.append(model)
    # # models.append(model.encoder)
    ####### loaded
    
    # Load the models
    models = []
    for i, path in enumerate(model_paths):
        # model_paths order: [pretrained MAE, continually pretrained, finetuned, ...]
        if i == 0:
            model = vit_pre
            # vit_enc, lora_params = apply_lora_to_vit(
            #     vit_enc,            # The ViT model
            #     rank=16,         # Rank of the decomposition
            #     lora_alpha=32,   # Scaling factor
            #     lora_dropout=0.1, # Dropout probability
            #     freeze_original=False, # Freeze original parameters
            # )

            # adapter = ViTAdapter(
            # img_size=args.crop_spatial_size,
            # patch_size=(16, 32, 32),
            # embed_dim=768,)

            # model = UperNet(encoder=vit_enc, in_channels=[768] * 4,
            #                 out_channels=args.out_channels, adapter=adapter,
            #                 out_indices=[3, 5, 7, 11],dropout_rate=0.1,
            #                 )
            
            # model = model.encoder

        elif i == 1:
            model = vit_cont
        else:
            # finetuned encoders are loaded into a plain ViT (as in the paper runs)
            model = vit_pre_ft if i == 2 else vit_base_patch16_dec512d8b(in_chans=args.in_channels)
            
        print(f"Loading model {i+1}/{len(model_paths)}: {os.path.basename(path)}")
        # checkpoint = torch.load(path, map_location='cpu')
        # # if "output_mae" not in path:
        # # Adjust keys to remove 'encoder.' prefix if present
        # checkpoint['model'] = {k.replace('encoder.', ''): v for k, v in checkpoint['model'].items()}
        # # print("Keys in the weights file:")
        # # print(checkpoint["model"].keys())
        # msg = model.load_state_dict(checkpoint['model'], strict=False)

        ######################
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint['model'] = {k.replace('encoder.', ''): v for k, v in checkpoint['model'].items()}
        state_dict = checkpoint["model"]
        # Adapt the patch embedding weights from 3 to 1 channel by averaging
        if "patch_embed.proj.weight" in state_dict:
            orig_weight = state_dict["patch_embed.proj.weight"]  # [768, 3, 16, 32, 32]
            if orig_weight.shape[1] != model.patch_embed.proj.weight.shape[1]:
                print(f"Adapting patch_embed.proj.weight from {orig_weight.shape} to {model.patch_embed.proj.weight.shape}")
                # Average over the input channels dimension (dim=1)
                state_dict["patch_embed.proj.weight"] = orig_weight.mean(dim=1, keepdim=True)

        msg = model.load_state_dict(state_dict, strict=False)
        print("missing keys:", msg.missing_keys)
        ######################

        # print("missing keys:", msg.missing_keys)
        # print(model)
        # Move model to the same device as input_batch
        model = model.to(device)
        model.eval()
        print(f"Model {i+1} moved to {device}")
        
        models.append(model)
    
    # Get outputs from each layer for all models
    all_model_features = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            print(f"Processing model {i+1}/{len(models)}")
            try:
                # Process the entire batch at once
                print(f"Input batch shape: {input_batch.shape}")
                
                # if sliding_window:
                #     def model_wrapper(x):
                #         return model(x, ret_hids=True)[2]
                #     hidden_states = sliding_window_inference(
                #         inputs=input_batch,
                #         roi_size=args.crop_spatial_size,  # Make sure this is properly defined
                #         sw_batch_size=4,
                #         predictor=model_wrapper,
                #         overlap=0.5
                #     )
                # else:
                #     outputs = model(input_batch, ret_hids=True)
                outputs = model(input_batch, ret_hids=True)
                # Unpack the tuple correctly
                _, _, hidden_states = outputs
                
                print(f"Model {i+1} returned {len(hidden_states)} hidden states")
                
                # Select specific layers
                indices = [3, 5, 7, 11]
                available_indices = [idx for idx in indices if idx < len(hidden_states)]
                
                if len(available_indices) < len(indices):
                    print(f"WARNING: Model {i+1} only has {len(hidden_states)} layers. Using indices {available_indices}")
                
                selected_features = [hidden_states[i] for i in available_indices]
                
                # If we have fewer layers than expected, repeat the last layer
                while len(selected_features) < len(indices):
                    print(f"Padding with repeated layers for model {i+1}")
                    selected_features.append(selected_features[-1])
                    
                all_model_features.append(selected_features)
            except Exception as e:
                print(f"Error processing model {i+1}: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    # Compute CKA similarity between models
    num_models = len(models)
    num_layers = len(indices)  # We want 4 layers
    
    # Compare same layers across different models
    cka_results = np.zeros((num_models, num_models, num_layers))
    
    # Set diagonal to 1.0 (same model compared to itself)
    for model_i in range(num_models):
        for l in range(num_layers):
            cka_results[model_i, model_i, l] = 1.0
    
    # Compare different models
    for layer_idx in range(num_layers):
        for model_i in range(num_models):
            for model_j in range(model_i+1, num_models):  # Only compute upper triangle
                # Get features for this layer from both models
                features_i = all_model_features[model_i][layer_idx]
                features_j = all_model_features[model_j][layer_idx]
                
                # Compute CKA across the batch dimension
                batch_cka = 0.0
                batch_size = features_i.shape[0]
                
                for b in range(batch_size):
                    # Extract single sample features
                    feat_i_sample = features_i[b].cpu().numpy()
                    feat_j_sample = features_j[b].cpu().numpy()
                    
                    # Compute CKA for this sample
                    sample_cka = centered_kernel_alignment(feat_i_sample, feat_j_sample)
                    batch_cka += sample_cka
                
                # Average CKA across the batch
                avg_cka = batch_cka / batch_size
                # Fill both positions in the matrix (symmetric)
                cka_results[model_i, model_j, layer_idx] = avg_cka
                cka_results[model_j, model_i, layer_idx] = avg_cka
    
    return cka_results


def compare_vit_encoders_with_cka_batch_SR(args, model_paths, input_batch, input_HR, sliding_window=False):
    """
    Compare three ViT encoders using CKA with a batch of inputs.
    """
    # Determine the device from input_batch
    device = input_batch.device
    print(f"Input batch is on device: {device}")
    
    vit_pre, vit_cont, vit_ft, _ = load_vit_models(args, SR=True)

    ####loading models
    # models = []
    # # define the model
    # device = torch.device("cuda")
    # model, optimizer = buidl_model(args=args, device=device) #pretrained weights loaded
    # model.eval()
    # filepath_best = os.path.join(args.output_dir, "best.pth.tar")
    # # filepath_best = os.path.join("output_seg_lesion_pp_lora_dec799epoch/Promis/profound_vit/lora/2021", "best.pth.tar") # custom path
    # model.load_state_dict(torch.load(filepath_best)["model"])
    # print("loaded for test from ", filepath_best)
    # models.append(model)
    # # models.append(model.encoder)
    ####### loaded
    
    # Load the models
    models = []
    for i, path in enumerate(model_paths):
        # model_paths order: [pretrained MAE, continually pretrained, finetuned, ...]
        if i == 0:
            model = vit_pre
            # vit_enc, lora_params = apply_lora_to_vit(
            #     vit_enc,            # The ViT model
            #     rank=16,         # Rank of the decomposition
            #     lora_alpha=32,   # Scaling factor
            #     lora_dropout=0.1, # Dropout probability
            #     freeze_original=False, # Freeze original parameters
            # )

            # adapter = ViTAdapter(
            # img_size=args.crop_spatial_size,
            # patch_size=(16, 32, 32),
            # embed_dim=768,)

            # model = UperNet(encoder=vit_enc, in_channels=[768] * 4,
            #                 out_channels=args.out_channels, adapter=adapter,
            #                 out_indices=[3, 5, 7, 11],dropout_rate=0.1,
            #                 )
            
            # model = model.encoder

        elif i == 1:
            model = vit_cont
        else:
            model = vit_ft

            
        print(f"Loading model {i+1}/{len(model_paths)}: {os.path.basename(path)}")
        # checkpoint = torch.load(path, map_location='cpu')
        # # if "output_mae" not in path:
        # # Adjust keys to remove 'encoder.' prefix if present
        # checkpoint['model'] = {k.replace('encoder.', ''): v for k, v in checkpoint['model'].items()}
        # # print("Keys in the weights file:")
        # # print(checkpoint["model"].keys())
        # msg = model.load_state_dict(checkpoint['model'], strict=False)

        ######################
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint['model'] = {k.replace('encoder.', ''): v for k, v in checkpoint['model'].items()}
        state_dict = checkpoint["model"]
        # Adapt the patch embedding weights from 3 to 1 channel by averaging
        if "patch_embed.proj.weight" in state_dict:
            orig_weight = state_dict["patch_embed.proj.weight"]  # [768, 3, 16, 32, 32]
            if orig_weight.shape[1] != model.patch_embed.proj.weight.shape[1]:
                print(f"Adapting patch_embed.proj.weight from {orig_weight.shape} to {model.patch_embed.proj.weight.shape}")
                # Average over the input channels dimension (dim=1)
                state_dict["patch_embed.proj.weight"] = orig_weight.mean(dim=1, keepdim=True)

        msg = model.load_state_dict(state_dict, strict=False)
        print("missing keys:", msg.missing_keys)
        ######################

        # print("missing keys:", msg.missing_keys)
        # print(model)
        # Move model to the same device as input_batch
        model = model.to(device)
        model.eval()
        print(f"Model {i+1} moved to {device}")
        
        models.append(model)
    
    # Get outputs from each layer for all models
    all_model_features = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            print(f"Processing model {i+1}/{len(models)}")
            try:
                # Process the entire batch at once
                print(f"LR Input batch shape: {input_batch.shape}")
                print(f"HR Input batch shape: {input_HR.shape}")

                
                # if sliding_window:
                #     def model_wrapper(x):
                #         return model(x, ret_hids=True)[2]
                #     hidden_states = sliding_window_inference(
                #         inputs=input_batch,
                #         roi_size=args.crop_spatial_size,  # Make sure this is properly defined
                #         sw_batch_size=4,
                #         predictor=model_wrapper,
                #         overlap=0.5
                #     )
                # else:
                #     outputs = model(input_batch, ret_hids=True)
                if i == 2:  # Assuming the 3rd model is finetuned 
                    outputs = model(input_batch, ret_hids=True)
                else:
                    outputs = model(input_HR, ret_hids=True)
                # Unpack the tuple correctly
                _, _, hidden_states = outputs
                
                print(f"Model {i+1} returned {len(hidden_states)} hidden states")
                
                # Select specific layers
                indices = [3, 5, 7, 11]
                available_indices = [idx for idx in indices if idx < len(hidden_states)]
                
                if len(available_indices) < len(indices):
                    print(f"WARNING: Model {i+1} only has {len(hidden_states)} layers. Using indices {available_indices}")
                
                selected_features = [hidden_states[i] for i in available_indices]
                
                # If we have fewer layers than expected, repeat the last layer
                while len(selected_features) < len(indices):
                    print(f"Padding with repeated layers for model {i+1}")
                    selected_features.append(selected_features[-1])
                    
                all_model_features.append(selected_features)
            except Exception as e:
                print(f"Error processing model {i+1}: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    # Compute CKA similarity between models
    num_models = len(models)
    num_layers = len(indices)  # We want 4 layers
    
    # Compare same layers across different models
    cka_results = np.zeros((num_models, num_models, num_layers))
    
    # Set diagonal to 1.0 (same model compared to itself)
    for model_i in range(num_models):
        for l in range(num_layers):
            cka_results[model_i, model_i, l] = 1.0
    
    # Compare different models
    for layer_idx in range(num_layers):
        for model_i in range(num_models):
            for model_j in range(model_i+1, num_models):  # Only compute upper triangle
                # Get features for this layer from both models
                features_i = all_model_features[model_i][layer_idx]
                features_j = all_model_features[model_j][layer_idx]
                
                # Compute CKA across the batch dimension
                batch_cka = 0.0
                batch_size = features_i.shape[0]
                
                for b in range(batch_size):
                    # Extract single sample features
                    feat_i_sample = features_i[b].cpu().numpy()
                    feat_j_sample = features_j[b].cpu().numpy()

                    # RESIZE FEATURES IF DIMENSIONS DON'T MATCH
                    if feat_i_sample.shape != feat_j_sample.shape:
                        print(f"Resizing features from shape {feat_i_sample.shape} to {feat_j_sample.shape}")
                        
                        # Determine which feature map needs resizing
                        if np.prod(feat_i_sample.shape) < np.prod(feat_j_sample.shape):
                            # Resize i to match j
                            feat_i_sample = resize_feature_map(feat_i_sample, feat_j_sample.shape)
                        else:
                            # Resize j to match i
                            feat_j_sample = resize_feature_map(feat_j_sample, feat_i_sample.shape)
                    
                    # Compute CKA for this sample
                    sample_cka = centered_kernel_alignment(feat_i_sample, feat_j_sample)
                    batch_cka += sample_cka
                
                # Average CKA across the batch
                avg_cka = batch_cka / batch_size
                # Fill both positions in the matrix (symmetric)
                cka_results[model_i, model_j, layer_idx] = avg_cka
                cka_results[model_j, model_i, layer_idx] = avg_cka
    
    return cka_results