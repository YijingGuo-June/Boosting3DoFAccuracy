import time
import torch
from torchvision import transforms
import gc
from thop import profile
from DINOv2_featup import DINOv2Featurizer
from DINOv2_featup import Encoder as DINOv2Encoder
from VGG import VGGUnet, VGGUnet_G2S, Encoder, Decoder, Decoder2, Decoder4, VGGUnetTwoDec
from SLR import add_extra_weights, ScaledLowRankAdapter, ScaledLowRankConvAdapter, ScaledLowRankConfigTimmViT
from DINOv2 import DINOv2
# def test_model(model_type="original"):
#     # 清空GPU缓存
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # 重置CUDA内存统计
#     torch.cuda.reset_peak_memory_stats()
    
#     device = torch.device('cuda')
    
#     # 加载相应的模型
#     if model_type == "original":
#         print("\n测试原始DINOv2模型...")
#         model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', trust_repo=True)
#     else:
#         print("\n测试添加适配器后的DINOv2模型...")
#         model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', trust_repo=True)
#         config = ScaledLowRankConfigTimmViT()
#         add_extra_weights(
#             model.model,
#             config,
#             adapter=ScaledLowRankAdapter,
#             conv_adapter=ScaledLowRankConvAdapter,
#             trainable=True,
#             only_scaler_trainable=False
#         )
    
#     model = model.to(device)
    
#     # 创建输入
#     x = torch.randn(1, 3, 518, 518).to(device)
    
#     # 计算参数量
#     params = sum(p.numel() for p in model.parameters())
#     print(f"参数量: {params:,}")
    
#     # 测量推理时间
#     def measure_time(model, x, runs=100):
#         torch.cuda.synchronize()
#         start = time.time()
#         with torch.no_grad():
#             for _ in range(runs):
#                 _ = model(x)
#                 torch.cuda.synchronize()
#         return (time.time() - start) / runs * 1000  # ms
    
#     inference_time = measure_time(model, x)
#     print(f"推理时间: {inference_time:.2f}ms")
    
#     # 测量内存使用
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         _ = model(x)
#     memory_used = torch.cuda.max_memory_allocated() / 1024**2
#     print(f"内存使用: {memory_used:.2f}MB")
    
#     # 清理内存
#     del model
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return params, inference_time, memory_used

# def compare_versions():
#     print("开始性能比较测试...")
    
#     # 测试原始模型
#     orig_params, orig_time, orig_memory = test_model("original")
    
#     # 测试适配器模型
#     adapted_params, adapted_time, adapted_memory = test_model("adapted")
    
#     # 打印比较结果
#     print("\n=== 最终比较结果 ===")
#     print(f"参数量比较:")
#     print(f"原始DINOv2: {orig_params:,}")
#     print(f"加适配器后: {adapted_params:,}")
#     print(f"增加量: {adapted_params - orig_params:,} ({(adapted_params/orig_params - 1)*100:.2f}%)")
    
#     print(f"\n推理时间比较:")
#     print(f"原始DINOv2: {orig_time:.2f}ms")
#     print(f"加适配器后: {adapted_time:.2f}ms")
#     print(f"增加量: {adapted_time - orig_time:.2f}ms ({(adapted_time/orig_time - 1)*100:.2f}%)")
    
#     print(f"\n内存使用比较:")
#     print(f"原始DINOv2: {orig_memory:.2f}MB")
#     print(f"加适配器后: {adapted_memory:.2f}MB")
#     print(f"增加量: {adapted_memory - orig_memory:.2f}MB ({(adapted_memory/orig_memory - 1)*100:.2f}%)")

# if __name__ == "__main__":
#     compare_versions()


# def test_feature_extractor(model_type="dinov2"):
#     # 清空GPU缓存
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # 重置CUDA内存统计
#     torch.cuda.reset_peak_memory_stats()
    
#     device = torch.device('cuda')
    
#     # 加载相应的模型
#     if model_type == "dinov2":
#         print("\n测试DINOv2Featurizer...")
#         model = DINOv2Featurizer()
#     else:
#         print("\n测试VGGUnet...")
#         model = VGGUnet(level=3)  # 使用相同的level参数
    
#     model = model.to(device)
    
#     # 创建输入
#     x = torch.randn(1, 3, 512, 512).to(device)  # 使用512x512的输入
    
#     # 计算参数量
#     params = sum(p.numel() for p in model.parameters())
#     print(f"参数量: {params:,}")
    
#     # 测量推理时间
#     def measure_time(model, x, runs=100):
#         torch.cuda.synchronize()
#         start = time.time()
#         with torch.no_grad():
#             for _ in range(runs):
#                 _ = model(x)
#                 torch.cuda.synchronize()
#         return (time.time() - start) / runs * 1000  # ms
    
#     inference_time = measure_time(model, x)
#     print(f"推理时间: {inference_time:.2f}ms")
    
#     # 测量内存使用
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         _ = model(x)
#     memory_used = torch.cuda.max_memory_allocated() / 1024**2
#     print(f"内存使用: {memory_used:.2f}MB")
    
#     # 清理内存
#     del model
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return params, inference_time, memory_used

# def compare_feature_extractors():
#     print("开始特征提取器性能比较测试...")
    
#     # 测试DINOv2
#     dino_params, dino_time, dino_memory = test_feature_extractor("dinov2")
    
#     # 测试VGGUnet
#     vgg_params, vgg_time, vgg_memory = test_feature_extractor("vgg")
    
#     # 打印比较结果
#     print("\n=== 最终比较结果 ===")
#     print(f"参数量比较:")
#     print(f"DINOv2Featurizer: {dino_params:,}")
#     print(f"VGGUnet: {vgg_params:,}")
#     print(f"差异: {dino_params - vgg_params:,} ({(dino_params/vgg_params - 1)*100:.2f}%)")
    
#     print(f"\n推理时间比较:")
#     print(f"DINOv2Featurizer: {dino_time:.2f}ms")
#     print(f"VGGUnet: {vgg_time:.2f}ms")
#     print(f"差异: {dino_time - vgg_time:.2f}ms ({(dino_time/vgg_time - 1)*100:.2f}%)")
    
#     print(f"\n内存使用比较:")
#     print(f"DINOv2Featurizer: {dino_memory:.2f}MB")
#     print(f"VGGUnet: {vgg_memory:.2f}MB")
#     print(f"差异: {dino_memory - vgg_memory:.2f}MB ({(dino_memory/vgg_memory - 1)*100:.2f}%)")

# if __name__ == "__main__":
#     compare_feature_extractors()

# def test_dinov2_steps():
#     # 清空GPU缓存
#     torch.cuda.empty_cache()
#     gc.collect()
#     torch.cuda.reset_peak_memory_stats()
    
#     device = torch.device('cuda')
#     model = DINOv2Featurizer().to(device)
#     x = torch.randn(1, 3, 512, 512).to(device)
    
#     def measure_time(func, runs=100):
#         torch.cuda.synchronize()
#         start = time.time()
#         with torch.no_grad():
#             for _ in range(runs):
#                 _ = func()
#                 torch.cuda.synchronize()
#         return (time.time() - start) / runs * 1000  # ms
    
#     print("\n=== DINOv2Featurizer 步骤时间分析 ===")
    
#     # 1. 测试padding
#     time_padding = measure_time(lambda: model.center_padding(x))
#     print(f"1. Padding时间: {time_padding:.2f}ms")
    
#     # 2. 测试DINOv2主干网络
#     padded_x = model.center_padding(x)
#     time_backbone = measure_time(lambda: model.model(padded_x))
#     print(f"2. DINOv2主干网络时间: {time_backbone:.2f}ms")
    
#     # 3. 测试裁剪
#     hr_feats = model.model(padded_x)
#     pad_h = (model.patch_size - x.shape[2] % model.patch_size) % model.patch_size
#     pad_w = (model.patch_size - x.shape[3] % model.patch_size) % model.patch_size
#     pad_t = pad_h // 2
#     pad_l = pad_w // 2
#     time_crop = measure_time(lambda: hr_feats[:, :, pad_t:pad_t+512, pad_l:pad_l+512])
#     print(f"3. 特征裁剪时间: {time_crop:.2f}ms")
    
#     # 4. 测试下采样和置信度预测
#     hr_feats_cropped = hr_feats[:, :, pad_t:pad_t+512, pad_l:pad_l+512]
    
#     # Level 0
#     time_down0 = measure_time(lambda: model.downsamplers[0](hr_feats_cropped, None))
#     feat0 = model.downsamplers[0](hr_feats_cropped, None)
#     time_conf0 = measure_time(lambda: model.conf_heads[0](feat0))
#     print(f"4.1 Level 0 下采样时间: {time_down0:.2f}ms")
#     print(f"4.1 Level 0 置信度预测时间: {time_conf0:.2f}ms")
    
#     # Level 1
#     time_down1 = measure_time(lambda: model.downsamplers[1](hr_feats_cropped, None))
#     feat1 = model.downsamplers[1](hr_feats_cropped, None)
#     time_conf1 = measure_time(lambda: model.conf_heads[1](feat1))
#     print(f"4.2 Level 1 下采样时间: {time_down1:.2f}ms")
#     print(f"4.2 Level 1 置信度预测时间: {time_conf1:.2f}ms")
    
#     # Level 2
#     time_down2 = measure_time(lambda: model.downsamplers[2](hr_feats_cropped, None))
#     feat2 = model.downsamplers[2](hr_feats_cropped, None)
#     time_conf2 = measure_time(lambda: model.conf_heads[2](feat2))
#     print(f"4.3 Level 2 下采样时间: {time_down2:.2f}ms")
#     print(f"4.3 Level 2 置信度预测时间: {time_conf2:.2f}ms")
    
#     # 总时间
#     total_time = measure_time(lambda: model(x))
#     print(f"\n总推理时间: {total_time:.2f}ms")
    
#     # 计算各部分占比
#     backbone_percent = time_backbone / total_time * 100
#     downsample_percent = (time_down0 + time_down1 + time_down2) / total_time * 100
#     conf_percent = (time_conf0 + time_conf1 + time_conf2) / total_time * 100
#     other_percent = 100 - backbone_percent - downsample_percent - conf_percent
    
#     print("\n时间占比:")
#     print(f"DINOv2主干网络: {backbone_percent:.1f}%")
#     print(f"下采样操作: {downsample_percent:.1f}%")
#     print(f"置信度预测: {conf_percent:.1f}%")
#     print(f"其他操作: {other_percent:.1f}%")
    
#     # 清理内存
#     del model
#     torch.cuda.empty_cache()
#     gc.collect()

# if __name__ == "__main__":
#     test_dinov2_steps()


# def test_model(model_type="original"):
#     # 清空GPU缓存
#     torch.cuda.empty_cache()
#     gc.collect()
#     torch.cuda.reset_peak_memory_stats()
    
#     device = torch.device('cuda')
    
#     # 加载相应的模型
#     if model_type == "original":
#         print("\n测试原始DINOv2模型...")
#         model = DINOv2()
#     else:
#         print("\n测试FeatUp版本DINOv2模型...")
#         model = DINOv2Featurizer()
    
#     model = model.to(device)
    
#     # 创建输入
#     x = torch.randn(1, 3, 512, 512).to(device)
    
#     # 计算参数量
#     params = sum(p.numel() for p in model.parameters())
#     print(f"参数量: {params:,}")
    
#     # 测量推理时间（无梯度）
#     def measure_time_no_grad(model, x, runs=100):
#         torch.cuda.synchronize()
#         start = time.time()
#         with torch.no_grad():
#             for _ in range(runs):
#                 _ = model(x)
#                 torch.cuda.synchronize()
#         return (time.time() - start) / runs * 1000  # ms
    
#     # 测量推理时间（有梯度）
#     def measure_time_with_grad(model, x, runs=10):
#         torch.cuda.synchronize()
#         start = time.time()
#         for _ in range(runs):
#             output = model(x)
#             if isinstance(output, tuple):
#                 loss = sum([feat.mean() for feat in output[0]])
#             else:
#                 loss = output.mean()
#             loss.backward()
#             torch.cuda.synchronize()
#         return (time.time() - start) / runs * 1000  # ms
    
#     no_grad_time = measure_time_no_grad(model, x)
#     print(f"无梯度推理时间: {no_grad_time:.2f}ms")
    
#     # 重置内存统计
#     torch.cuda.reset_peak_memory_stats()
#     with_grad_time = measure_time_with_grad(model, x)
#     print(f"有梯度推理时间: {with_grad_time:.2f}ms")
    
#     # 测量内存使用
#     memory_used = torch.cuda.max_memory_allocated() / 1024**2
#     print(f"内存使用: {memory_used:.2f}MB")
    
#     # 清理内存
#     del model
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return params, no_grad_time, with_grad_time, memory_used

# def compare_models():
#     print("开始模型比较测试...")
    
#     # 测试原始DINOv2
#     orig_params, orig_no_grad, orig_with_grad, orig_memory = test_model("original")
    
#     # 测试FeatUp版本
#     featup_params, featup_no_grad, featup_with_grad, featup_memory = test_model("featup")
    
#     # 打印比较结果
#     print("\n=== 最终比较结果 ===")
#     print(f"参数量比较:")
#     print(f"原始DINOv2: {orig_params:,}")
#     print(f"FeatUp版本: {featup_params:,}")
#     print(f"差异: {featup_params - orig_params:,} ({(featup_params/orig_params - 1)*100:.2f}%)")
    
#     print(f"\n无梯度推理时间比较:")
#     print(f"原始DINOv2: {orig_no_grad:.2f}ms")
#     print(f"FeatUp版本: {featup_no_grad:.2f}ms")
#     print(f"差异: {featup_no_grad - orig_no_grad:.2f}ms ({(featup_no_grad/orig_no_grad - 1)*100:.2f}%)")
    
#     print(f"\n有梯度推理时间比较:")
#     print(f"原始DINOv2: {orig_with_grad:.2f}ms")
#     print(f"FeatUp版本: {featup_with_grad:.2f}ms")
#     print(f"差异: {featup_with_grad - orig_with_grad:.2f}ms ({(featup_with_grad/orig_with_grad - 1)*100:.2f}%)")
    
#     print(f"\n内存使用比较:")
#     print(f"原始DINOv2: {orig_memory:.2f}MB")
#     print(f"FeatUp版本: {featup_memory:.2f}MB")
#     print(f"差异: {featup_memory - orig_memory:.2f}MB ({(featup_memory/orig_memory - 1)*100:.2f}%)")

# if __name__ == "__main__":
#     compare_models()




# def compare_base_models():
#     print("=== 基础模型比较 ===")
    
#     # 清空GPU缓存
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # 加载原始DINOv2
#     print("\n加载原始DINOv2...")
#     original_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
#     orig_params = sum(p.numel() for p in original_model.parameters())
    
#     # 打印每层参数
#     print("\nDINOv2原始模型结构:")
#     for name, param in original_model.named_parameters():
#         print(f"{name}: {param.numel():,} 参数")
    
#     # 清理第一个模型
#     del original_model
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # 加载FeatUp版本
#     print("\n加载FeatUp版本...")
#     featup_model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', trust_repo=True)
#     featup_params = sum(p.numel() for p in featup_model.parameters())
    
#     print("\nFeatUp模型结构:")
#     for name, param in featup_model.named_parameters():
#         print(f"{name}: {param.numel():,} 参数")
#     print(featup_model)
#     print(featup_model.model)
    
#     # 比较结果
#     print("\n=== 参数量比较 ===")
#     print(f"原始DINOv2: {orig_params:,} 参数")
#     print(f"FeatUp版本: {featup_params:,} 参数")
#     print(f"差异: {featup_params - orig_params:,} 参数")
#     print(f"增加比例: {(featup_params/orig_params - 1)*100:.2f}%")
    
#     # 清理第二个模型
#     del featup_model
#     torch.cuda.empty_cache()
#     gc.collect()

# if __name__ == "__main__":
#     compare_base_models()


# def inspect_model_structure():
#     print("=== 检查FeatUp模型结构 ===")
    
#     # 加载FeatUp模型
#     model = torch.hub.load("mhamilton723/FeatUp", 'dinov2', trust_repo=True)
    
#     # 1. 打印模型结构
#     print("\n1. 完整模型结构:")
#     print(model)
    
#     # 2. 打印model属性
#     print("\n2. model属性:")
#     print(model.model)
    
#     # 3. 检查参数名和它们的形状
#     print("\n3. 参数名和形状:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.shape}")
    
#     # 4. 尝试不同的访问方式
#     print("\n4. 尝试访问第一个参数:")
#     # 获取第一个参数的名字
#     first_param_name = next(iter(model.named_parameters()))[0]
#     print(f"第一个参数名: {first_param_name}")
    
#     # 5. 打印模型的属性
#     print("\n5. 模型的所有属性:")
#     print(dir(model))

# if __name__ == "__main__":
#     inspect_model_structure()


import torch
from DINOv2_MAE import MAE_Sat

# 创建模型
mae = MAE_Sat(
    output="dense",
    return_multilayer=True,
    output_channels=112
)

mae.print_trainable_params()

# 准备输入数据
images = torch.randn(16, 3, 518, 518)

# 提取特征
with torch.no_grad():
    multi_scale_features, confidence_scores = mae(images)

print("Multi-scale features shape:", multi_scale_features[0].shape, multi_scale_features[1].shape, multi_scale_features[2].shape)
print("Confidence scores shape:", confidence_scores[0].shape, confidence_scores[1].shape, confidence_scores[2].shape)