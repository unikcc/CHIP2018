## 平安科技医疗文本匹配大赛 Rank8
### 模型说明
+ 基于ESIM稍加改造，只使用了char级别的信息，除了预训练char级别词向量外没有使用任何特征工程
+ 可惜没有太多精力，只做了单模型，本地十折交叉验证
### 运行程序
+ 将数据集放在./data/pingan下，如./data/pingan/char_embedding.csv
+ cd scritps
+ (可选) pip install requirements.txt
+ chmod +x step.sh
+ ./step.sh即可执行预处理，训练与验证，预测等全过程

### 其他
+ Believe your local cv!
