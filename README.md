# Orientation Human Loop + Baidu Manual Submit

这个开发分支包含两个隔离的部分：

1. 图片方向学习：PyTorch 学习“给定一张随机歪图，应该旋转多少才能恢复自然正立方向”。
2. 百度 URL 提交：读取状态 JSON，只提交顶层 urls；遇到第三方人机验证时等待人工完成，然后继续。

## 图片方向学习目标

模型不学习角度出现顺序，也不模仿鼠标抖动。核心监督数据：

- initial.png：初始歪图
- upright.png：人工确认后的正立图
- correction_angle：从 initial 恢复到 upright 所需的逆时针修正角

交互轨迹、滑块位置和耗时可作为元数据保留，但不直接进入视觉模型。

## 目录

predict_orientation.py
train_orientation.py
manual_submit.py
orientation_core/
data/
tests/

本地运行时目录 dataset、checkpoints、reports 默认不提交 Git。

## URL 状态与 Cookie

已经从你上传的状态 JSON 中提取并提交了一份脱敏运行状态：

data/baidu-submit-urls.json

它保留：
- root_sitemaps
- 顶层 urls
- index
- completed
- skipped
- current_url
- status

不会把 sitemap_cache、网络信息等运行环境数据写进仓库。

真实 Cookie 包含登录会话凭据，而当前仓库是公开仓库，所以不能把真实 Cookie 写入 Git 历史。

本地放置：

data/cookies.json

这个路径已经加入 .gitignore。仓库提供 data/cookies.example.json 作为格式模板。

## 采集一条自有测试图片样本

python predict_orientation.py collect --image initial.png --upright upright.png --angle 143.6

如果没有单独的 upright.png，也可以：

python predict_orientation.py collect --image initial.png --angle 143.6

程序会保留 initial.png、upright.png 和 meta.json。

达到 100 条后首训，随后 150、200、250... 每新增 50 条可继续训练。

## 训练

python train_orientation.py --dataset dataset

## 推理

python predict_orientation.py predict --image test.png --model checkpoints/best.pt --save-corrected reports/predicted.png

## 百度人工验证提交

如果要直接用仓库里的 URL 状态：

python manual_submit.py --state data/baidu-submit-urls.json --cookies data/cookies.json

规则：
- sitemap 只作为 URL 发现来源；
- 真正提交只消费状态 JSON 顶层 urls；
- 自动跳过 completed 和 skipped；
- 遇到人机验证只等待人工处理；
- 不采集、不训练、不自动求解第三方 CAPTCHA。
