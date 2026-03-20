import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
import matplotlib.pyplot as plt
import os

# 手动指定DejaVu Sans字体路径（用户目录下）
font_path = os.path.expanduser("~/.fonts/DejaVu/DejaVuSans.ttf")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 强制matplotlib加载指定字体，绕过系统查找
plt.rcParams['font.family'] = 'sans-serif'
# BILIBILI UP 魔傀面具
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = 'runs/runs/augadd3_root/two69-all2/weights/epoch171.pt'
    model = YOLO(model_path) # 选择训练好的权重路径
    result = model.val(data='ultralytics/cfg/datasets/augadd3.yaml', ##cocov1,aug
                        split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
                        imgsz=640,
                        batch=8,
                        iou=0.45,
                        conf=0.25,
                        # rect=False,
                        # save_json=True, # if you need to cal coco metrice
                        #project='runs/test',
                        name='',
                        )
    
    if model.task == 'detect': # 仅目标检测任务适用
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx], 
                                        f"{result.box.p[idx]:.3f}", 
                                        f"{result.box.r[idx]:.3f}", 
                                        f"{result.box.f1[idx]:.3f}", 
                                        f"{result.box.ap50[idx]:.3f}", 
                                        f"{result.box.all_ap[idx, 5]:.3f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.3f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.3f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.3f}", 
                                    f"{np.mean(result.box.f1[:length]):.3f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.3f}", 
                                    f"{np.mean(result.box.all_ap[:length, 5]):.3f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.3f}"
                                ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)