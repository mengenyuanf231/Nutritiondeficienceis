import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

# 定量指标相关导入
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange, multiply_tensor_with_cam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputLnSoftmaxTarget,  ClassifierOutputEntropy, ClassifierOutputReST
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from scipy.linalg import norm
from scipy import stats as STS
import torch.nn.functional as FF
from typing import List, Callable

# ===================== 定量指标核心函数 =====================
def complexity(saliency_map):
    """计算复杂度指标：归一化的激活图绝对值和（返回标量）"""
    if len(saliency_map.shape) == 2:
        saliency_map = saliency_map[np.newaxis, ...]
    # 修复：计算全局均值，确保返回标量
    complexity_val = abs(saliency_map).sum(axis=(1, 2)) / (saliency_map.shape[-1] * saliency_map.shape[-2])
    return float(np.mean(complexity_val))  # 核心修改：取均值并转标量

def coherency(A, explanation_map, attr_method, targets):
    """计算相干性指标：激活图与扰动后激活图的皮尔逊相关系数（返回标量）"""
    # 适配YOLO输入格式
    if isinstance(explanation_map, torch.Tensor):
        explanation_map = explanation_map.cpu().numpy()
    
    # 修复核心：正确获取模型设备（从模型参数中提取，而非直接访问model.device）
    try:
        device = next(attr_method.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    
    # 生成扰动后的激活图
    B = attr_method(torch.tensor(explanation_map, dtype=torch.float32).to(device), targets)
    if isinstance(B, list) and len(B) > 0:
        B = B[0] if isinstance(B[0], np.ndarray) else B[0].cpu().detach().numpy()
    
    # 维度展平（展平为一维数组，而非二维）
    Asq = A.flatten()  # 核心修改：全局展平
    Bsq = B.flatten()  # 核心修改：全局展平

    # 处理NaN/Inf和零方差情况（直接返回标量0）
    if np.any(np.isnan(Bsq)) or np.any(np.isnan(Asq)) or np.any(np.isinf(Bsq)) or np.any(np.isinf(Asq)):
        return 0.0, A, B
    elif np.std(Bsq) == 0 or np.std(Asq) == 0:
        return 0.0, A, B
    else:
        corr, _ = STS.pearsonr(Asq, Bsq)
        return float((corr + 1) / 2), A, B  # 返回标量

class ADCC:
    """ADCC综合指标计算类：融合Coherency、Complexity、Average Drop（全程返回标量）"""
    def __init__(self):
        self.perturbation = multiply_tensor_with_cam

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 metric_targets: List[Callable],
                 model: torch.nn.Module,
                 cam_method,
                 return_visualization=False):
        # 原始模型输出得分
        with torch.no_grad():
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            elif isinstance(outputs, list):
                outputs = outputs[0]
        
        # 提取置信度得分（确保返回标量）
        pred = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.65)[0]
        score = float(pred[:, 4].max().cpu().numpy()) if len(pred) > 0 else 0.0  # 标量

        # 生成扰动后的张量
        added_tensors = []
        deleted_tensors = []
        for i in range(cams.shape[0]):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            added_tensors.append(tensor.unsqueeze(0))
            tensor_delete = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(1-cam))
            tensor_delete = tensor_delete.to(input_tensor.device)
            deleted_tensors.append(tensor_delete.unsqueeze(0))
        
        added_tensors = torch.cat(added_tensors) if added_tensors else input_tensor
        deleted_tensors = torch.cat(deleted_tensors) if deleted_tensors else input_tensor

        # 扰动后的模型输出（提取标量得分）
        with torch.no_grad():
            outputs_after_added = model(added_tensors)
            outputs_after_deleted = model(deleted_tensors)
            
            if isinstance(outputs_after_added, tuple):
                outputs_after_added = outputs_after_added[0]
            if isinstance(outputs_after_deleted, tuple):
                outputs_after_deleted = outputs_after_deleted[0]
            
            pred_added = non_max_suppression(outputs_after_added, conf_thres=0.25, iou_thres=0.65)[0]
            pred_deleted = non_max_suppression(outputs_after_deleted, conf_thres=0.25, iou_thres=0.65)[0]
            
            score_after_added = float(pred_added[:, 4].max().cpu().numpy()) if len(pred_added) > 0 else 0.0  # 标量
            score_after_deleted = float(pred_deleted[:, 4].max().cpu().numpy()) if len(pred_deleted) > 0 else 0.0  # 标量

        # 计算各子指标（全程标量）
        score = max(score, 1e-8)  # 避免除零
        drop = max(0.0, (score - score_after_added) / score)  # Average Drop（标量）
        inc = 1.0 if score_after_added > score else 0.0  # IC（标量）
        dropindeletion = max(0.0, (score - score_after_deleted) / score)  # ADD（标量）
        com = complexity(cams)  # Complexity（标量）
        coh, _, _ = coherency(cams, added_tensors, cam_method, targets)  # Coherency（标量）

        # 计算ADCC（确保标量，避免除零）
        coh = max(coh, 1e-8)
        com = min(com, 0.9999)
        drop = min(drop, 0.9999)
        adcc = 3 / (1 / coh + 1 / (1 - com) + 1 / (1 - drop))
        adcc = float(adcc)  # 最终确认标量

        if return_visualization:
            return adcc, drop, coh, com, inc, dropindeletion, added_tensors, deleted_tensors
        else:
            return adcc, drop, coh, com, inc, dropindeletion

# ===================== YOLO热力图核心代码 =====================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

class CustomActivationsAndGradients(ActivationsAndGradients):
    """自定义激活和梯度提取类，避免析构报错"""
    def __init__(self, model, target_layers, reshape_transform):
        super().__init__(model, target_layers, reshape_transform)
    
    def release(self):
        try:
            for handle in self.handles:
                if handle is not None:
                    handle.remove()
            self.handles = []
        except Exception:
            pass

    def __del__(self):
        self.release()

    def post_process(self, result):
        if isinstance(result, tuple):
            result = result[0]
        if not hasattr(self.model, 'end2end'):
            self.model.end2end = False
        
        if not self.model.end2end:
            logits_ = result[:, 4:]
            boxes_ = result[:, :4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        else:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        post_result, pre_post_boxes = self.post_process(model_output)
        return [[post_result, pre_post_boxes]]

class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        # 限制循环次数，避免过度计算
        max_iter = max(1, int(post_result.size(0) * self.ratio))
        for i in trange(max_iter):
            if i >= post_result.size(0):
                break
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result) if result else torch.tensor(0.0).to(post_result.device)

# ===================== 可视化定量指标 =====================
def visualize_score(visualization, method_name, adcc, avg_drop, coherency, complexity_val, ic, add_val):
    """在热力图上绘制定量指标文本（确保输入为标量）"""
    # 最终兜底：强制转换为float标量
    adcc = float(adcc)
    avg_drop = float(avg_drop)
    coherency = float(coherency)
    complexity_val = float(complexity_val)
    ic = float(ic)
    add_val = float(add_val)

    # 绘制文本（调整字体大小，避免重叠）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color = (255, 255, 255)
    thickness = 1
    
    visualization = cv2.putText(visualization, f"Method: {method_name}", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"ADCC: {adcc:.5f}", (10, 40), font, font_scale, color, thickness, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"AD: {avg_drop:.5f}", (10, 60), font, font_scale, color, thickness, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Coherency: {coherency:.5f}", (10, 80), font, font_scale, color, thickness, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Complexity: {complexity_val:.5f}", (10, 100), font, font_scale, color, thickness, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"IC: {ic:.5f}", (10, 120), font, font_scale, color, thickness, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"ADD: {add_val:.5f}", (10, 140), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return visualization

# ===================== 保存定量指标到TXT =====================
def save_metrics_to_txt(save_path, img_name, metrics):
    """保存定量指标到txt文件（确保所有值为标量）"""
    metrics_dir = os.path.join(save_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    txt_path = os.path.join(metrics_dir, f"{img_name}_metrics.txt")
    
    # 强制转换为标量
    adcc = float(metrics['adcc'])
    avg_drop = float(metrics['avg_drop'])
    coherency = float(metrics['coherency'])
    complexity_val = float(metrics['complexity'])
    ic = float(metrics['ic'])
    add_val = float(metrics['add'])
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"# YOLO热力图定量评价指标\n")
        f.write(f"Method: {metrics['method']}\n")
        f.write(f"ADCC: {adcc:.5f}\n")
        f.write(f"Average Drop (AD): {avg_drop:.5f}\n")
        f.write(f"Coherency: {coherency:.5f}\n")
        f.write(f"Complexity: {complexity_val:.5f}\n")
        f.write(f"Increase (IC): {ic:.5f}\n")
        f.write(f"Drop in Deletion (ADD): {add_val:.5f}\n")

class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        self.device = torch.device(device)
        self.weight = weight
        self.method = method
        self.layer = layer
        self.backward_type = backward_type
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_box = show_box
        self.renormalize = renormalize
        
        # 加载模型
        ckpt = torch.load(weight, map_location=self.device)
        self.model_names = {0: 'Ca', 1: 'P', 2: 'FeM', 3: 'FeL', 4: 'PM'}
        print('类别名称:', self.model_names)
        self.model = attempt_load_weights(weight, self.device)
        self.model.info()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()
        
        # 初始化目标和CAM
        self.model.end2end = False if not hasattr(self.model, 'end2end') else self.model.end2end
        self.target = yolov8_target(backward_type, conf_threshold, ratio, self.model.end2end)
        target_layers = [self.model.model[l] for l in layer]
        
        # 初始化CAM方法
        cam_class = eval(method)
        self.cam_method = cam_class(model=self.model, target_layers=target_layers)
        self.activations_and_grads = CustomActivationsAndGradients(self.model, target_layers, None)
        self.cam_method.activations_and_grads = self.activations_and_grads
        
        # 初始化指标计算器
        self.adcc_metric = ADCC()
        self.metric_targets = [ClassifierOutputSoftmaxTarget(0)]
        
        # 颜色配置
        self.colors = np.random.uniform(0, 255, size=(len(self.model_names), 3)).astype(np.int32)
    
    def __del__(self):
        if hasattr(self, 'activations_and_grads'):
            self.activations_and_grads.release()
    
    def post_process(self, result):
        if isinstance(result, tuple):
            result = result[0]
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, box))
        # 确保坐标在图片范围内
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img.shape[1]-1, xmax)
        ymax = min(img.shape[0]-1, ymax)
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(c) for c in color), 2)
        # 调整文本位置，避免超出图片
        text_y = ymin - 5 if ymin - 5 > 0 else ymin + 15
        cv2.putText(img, str(name), (xmin, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(int(c) for c in color), 2, cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros_like(grayscale_cam, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(grayscale_cam.shape[1]-1, int(x2)), min(grayscale_cam.shape[0]-1, int(y2))
            if x1 < x2 and y1 < y2:  # 确保框有效
                renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        return show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    
    def process(self, img_path, save_path):
        # 1. 图片预处理
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return
        
        img_letterbox, ratio, pad = letterbox(img)
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        # 2. 生成热力图
        try:
            grayscale_cam = self.cam_method(input_tensor=tensor, targets=[self.target])
        except Exception as e:
            print(f"生成热力图失败 {img_path}: {str(e)[:200]}")
            return
        
        # 3. 计算定量指标
        adcc, avg_drop, coherency_val, complexity_val, ic, add_val = self.adcc_metric(
            input_tensor=tensor,
            cams=grayscale_cam,
            targets=[self.target],
            metric_targets=self.metric_targets,
            model=self.model,
            cam_method=self.cam_method
        )
        
        # 4. 生成热力图可视化
        grayscale_cam_single = grayscale_cam[0] if len(grayscale_cam.shape) > 2 else grayscale_cam
        cam_image = show_cam_on_image(img_float, grayscale_cam_single, use_rgb=True)
        
        # 5. 绘制检测框
        pred = self.model(tensor)
        if not self.model.end2end:
            pred = self.post_process(pred)
        else:
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred[0][pred[0, :, 4] > self.conf_threshold]
        
        if self.renormalize and len(pred) > 0:
            cam_image = self.renormalize_cam_in_bounding_boxes(
                pred[:, :4].cpu().detach().numpy(), 
                img_float, 
                grayscale_cam_single
            )
        if self.show_box and len(pred) > 0:
            for data in pred:
                data_np = data.cpu().detach().numpy()
                cls_idx = int(data_np[5])
                if cls_idx < len(self.colors):
                    cam_image = self.draw_detections(
                        data_np[:4], 
                        self.colors[cls_idx], 
                        f'{self.model_names[cls_idx]} {data_np[4]:.2f}', 
                        cam_image
                    )
        
        # 6. 标注定量指标
        cam_image = visualize_score(
            cam_image, 
            self.method, 
            adcc, avg_drop, coherency_val, complexity_val, ic, add_val
        )
        
        # 7. 保存结果
        cam_image_pil = Image.fromarray(cam_image)
        cam_image_pil.save(save_path)
        
        # 8. 保存指标
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        metrics = {
            'method': self.method,
            'adcc': adcc,
            'avg_drop': avg_drop,
            'coherency': coherency_val,
            'complexity': complexity_val,
            'ic': ic,
            'add': add_val
        }
        save_metrics_to_txt(os.path.dirname(save_path), img_name, metrics)
        
        print(f"处理完成：{img_path} -> {save_path}")
        print(f"定量指标：ADCC={adcc:.5f}, AD={avg_drop:.5f}, Coherency={coherency_val:.5f}")
    
    def __call__(self, img_path, save_path):
        # 初始化保存目录
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # 处理图片（单张/目录）
        if os.path.isdir(img_path):
            for img_file in os.listdir(img_path):
                img_file_path = os.path.join(img_path, img_file)
                if not os.path.isfile(img_file_path):
                    continue
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                save_file_path = os.path.join(save_path, img_file)
                self.process(img_file_path, save_file_path)
        else:
            save_file_path = os.path.join(save_path, 'result.png')
            self.process(img_path, save_file_path)
        
        print(f"\n所有图片处理完成！")
        print(f"热力图保存路径：{save_path}")
        print(f"定量指标保存路径：{os.path.join(save_path, 'metrics')}")

def get_params():
    """获取默认参数（可根据实际情况调整）"""
    params = {
        'weight': 'runs/runs/our_ImageSetsadd2/weights/best.pt', # 现在只需要指定权重即可,不需要指定cfg
        'device': 'cuda:0',
        'method': 'EigenCAM', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': [16,19,22], ##有效果的EigenCAM(v10s)、RandomCAM（没报错） #[19, 22, 25,28]；v10/11 [16,19,22] v8 [15,18,21];v5[17,20,23];v6[19,23,27]v5:17,20,23
        'backward_type': 'all', # class, box, all
        'conf_threshold': 0.20, # 0.2
        'ratio': 0.02, # 0.02-0.1
        'show_box': False, # 不需要绘制框请设置为False
        'renormalize': True # 需要把热力图限制在框内请设置为True
    }
    return params

# 主函数
if __name__ == '__main__':
    # 初始化并运行
    params = get_params()
    # 检查权重文件是否存在
   
    img_path = r'heat_imgs'
    
    heatmap_model = yolov8_heatmap(**params)
    heatmap_model(img_path, 'heat_result/our_ImageSetsadd2-best')