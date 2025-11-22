# run.py  —— 只新增两个超参，其它保持原样结构
import os
import argparse
import random
import numpy as np
import torch

from exp.exp_main import Exp_Main  # 保持你仓库里的入口不变

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='SST training/testing launcher')

    # ========= 原有常规参数（示例：保持你自己的为准；没有的可以忽略） =========
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model', type=str, default='SST')

    # 数据与路径（与你的工程一致）
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='data.csv')
    parser.add_argument('--features', type=str, default='M')     # M: multivariate, S: univariate, MS: multi to single
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')

    # 预测窗口
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # 模型宽度/层数（和你仓库参数一致即可）
    parser.add_argument('--enc_in', type=int, default=7)   # 变量个数 M
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fc_dropout', type=float, default=0.0)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--individual', action='store_true', default=False)

    # LWT（短程）补丁
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--local_ws', type=int, default=7)
    parser.add_argument('--padding_patch', type=str, default='end')

    # Mamba（长程）补丁
    parser.add_argument('--m_patch_len', type=int, default=64)
    parser.add_argument('--m_stride', type=int, default=32)
    parser.add_argument('--m_layers', type=int, default=3)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)

    # 其它常见开关
    parser.add_argument('--decomposition', action='store_true', default=False)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--revin', action='store_true', default=False)
    parser.add_argument('--affine', action='store_true', default=False)
    parser.add_argument('--subtract_last', action='store_true', default=False)
    parser.add_argument('--concat', action='store_true', default=True)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')

    # =====================【新增开始】=====================
    # 路由方式：global（原版，全局一对权重）| per_var（按变量，每变量一对权重）
    parser.add_argument('--router_type',
        type=str, default='global', choices=['global', 'per_var'],
        help="Router type: 'global' (original) or 'per_var' (per-variable router).")

    # 投影共享方式统一开关：
    # 0或1=共享；1<k<M=分组共享（k组）；k>=M=通道独立（每变量一套）
    parser.add_argument('--group_count',
        type=int, default=0,
        help="0/1=shared, 1<k<M=group-shared (k groups), k>=M=per-channel.")
    # =====================【新增结束】=====================

    args = parser.parse_args()

    # 可选打印，便于确认新开关已生效
    print(f"[SST switches] router_type={args.router_type} | group_count={args.group_count}")

    # 固定随机种子
    fix_seed(args.seed)

    # 训练/测试入口（保持你的工程原样）
    if args.is_training:
        for ii in range(args.itr):
            setting = f"{args.model}_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_gc{args.group_count}_rt{args.router_type}_itr{ii}"
            exp = Exp_Main(args)           # 由 exp/exp_main.py 构造并把 args 传下去
            print(f">>>>> start training: {setting}")
            exp.train(setting)
            print(f">>>>> testing: {setting}")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f"{args.model}_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_gc{args.group_count}_rt{args.router_type}_itr{ii}"
        exp = Exp_Main(args)
        print(f">>>>> only testing: {setting}")
        exp.test(setting)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
