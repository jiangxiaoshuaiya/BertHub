## ToyBert

**What's this：**

&emsp;&emsp;如其名，本项目的初衷是希望大家能轻易把玩bert及其一系列变体形式。这个项目以huggingface的开源项目transformers为基础。旨在搭建自然语言处理比赛较为通用的框架，达到稍作修改便能运行目的，帮助大家更快的上手。

**How to modify：**

+ 如果你想要定义自己的模型，请在model.py中修改；
+ 如果要改变数据处理方式，请在tokenization.py中修改；
+ 如果要添加评价指标，请在metric.py中修改；
+ 加载数据的方式见utils.py，请根据赛题文件形式选择或自行修改

### Example

👉🏿[2021海华AI挑战赛·中文阅读理解·技术组](https://www.biendata.xyz/competition/haihua_2021])baseline见example文件夹

![image-20210210174305778](https://imgchr.com/i/y0FzIs)

### 环境要求

显卡Tesla T4  14G ；torch == 1.7.0 ；transformers == 4.3.2；

```python
cd ../haihua
python run_classifier.py
```

随时接受issue，欢迎大家star ！！

未完待续，Updating ... ...

**Speed：**

--version 1.0.0-- 以2020 CCF 问答匹配赛题做简单demo，支持sequence classification

**添加海华阅读理解 QA baseline    2021年2月10日**

**随时接受issue，欢迎大家star ！！转载请带上链接，谢谢！！**

未完待续，Updating ... ...