# pairwise-letterbox-detect-crop

## 领域

游戏、图形与多媒体

## 主题

黑边检测与裁切

## 说明

阈值滞回、逐帧跟踪与显示空间裁切。

初始状态下 `media/letterbox_detect_crop.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
