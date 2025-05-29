--数据集
数据集保存在data文件夹，路径格式为：
--data
	|--train
		|--HR
		|--LR
	|--test
		|--HR
		|--LR
HR与LR中的图片名称要一致。请不要创建LR文件夹！train.py会自动创建LR！

--使用方式
训练：python train.py --epochs 4000 --batch_size 32  --pre_load True（训练结果保存在train文件夹下）
测试：python compare.py（测试结果保存在test文件夹下）
更多参数详见train.py和compare.py。

--请先安装相应的库
pip install -r requirements.txt

--历史训练结果
历史训练结果保存在result文件夹

--权重文件
weight文件夹下有两个权重文件，weight_new.pt是论文中使用的权重，其对应训练结果在result文件夹下
