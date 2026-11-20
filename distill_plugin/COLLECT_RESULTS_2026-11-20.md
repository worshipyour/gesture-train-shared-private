# 2026-11-20 只读汇总学生蒸馏与对照结果

在已有训练环境、仓库根目录执行：

```bash
python distill_plugin/collect_results_2026-11-20.py \
  --fgd-run runs/2026-11-19_191837_fgd_deployment
```

工具不启动训练、不使用GPU、不修改任何结果文件，也不导入torch或反序列化模型。它读取FGD的summary/status/opt/results文件，并在原工程gesture_train_shared/runs和当前仓库runs中查找对照。

比较使用同一初始化权重、学生模型/data/hyp配置内容、epochs、batch、seed、输入参数和主要训练开关。检查基于保存的opt.yaml和当前可读取的配置/初始化文件，无法事后证明图像标签或源码历史完全相同。缺失字段、文件缺失、恢复训练或训练预算不足会标为未匹配，不能因为没有通过匹配就断言历史对照从未运行。

验证指标优先读取summary中保存的精确best权重指标；旧工程没有summary时，回退到results.csv按0.1*mAP50+0.9*mAP50–95选择的行，明确注明CSV四舍五入且未重测best.pt。F1*由平均P/R推导。恰好一组配置匹配时输出FGD减对照的正负百分点变化；多组匹配时全部列出，不按精度挑选，需明确指定：

```bash
python distill_plugin/collect_results_2026-11-20.py \
  --fgd-run runs/2026-11-19_191837_fgd_deployment \
  --baseline-run /absolute/path/to/baseline_run
```

若旧工程在其他位置，用--original-project指定。没有匹配时，先核对输出的差异与未知项；确实没有可用同条件无KD对照时，再在空闲GPU运行已提供的--mode baseline，使用与FGD同一epoch76.pt、学生YAML/data/hyp、seed0、batch128和200轮。不能用只验证初始epoch76.pt代替同预算继续训练的无KD对照。

工具输出只涵盖验证结果及其配置核对，不包含AP95、正式测试或推理延迟，不宣称蒸馏一定有效。新增5项回归覆盖fitness选行、精确summary优先、随机种子不匹配、正负变化和多组对照不择优挑选；全部18项检查通过。
