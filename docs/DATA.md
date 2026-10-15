# 数据来源和处理协议

- 官方页面：https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
- 镜像：https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection
- 固定提交：351bee21b56bbf97a1e1d18262bceb202eb7ef41
- [下载固定提交ZIP](https://codeload.github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection/zip/351bee21b56bbf97a1e1d18262bceb202eb7ef41)
- ZIP SHA256：e0f360b032342399d18ca14328e9eff905e7ac18cb5b1bc91f1fd20512e267d6

下载后只需复制图片和XML目录，不需要运行镜像仓库代码。准备后的结构：

```text
steel-defect-detection/
  NEU-DET原始镜像/
    IMAGES/
    ANNOTATIONS/
    Validation_Images/
    Validation_Annotations/
  prepare_day14_data.py
```

原始镜像的文件夹名称不是本实验的最终划分。脚本合并这四个目录的数据，重新固定分为1260/270/270张。

六类图片各300张，共1800张、4189个目标。按每个object/name取类别，不用文件名前缀替代目标类别；文件名前缀仅用于分层。174份XML内filename字段与真实名称不完全一致，配对使用XML文件名主体。

patches_101.jpg与patches_105.jpg字节完全相同但标注略有差别，保留两张并分入同一集合。相同SHA256组不跨集合；不能据此声称排除了近似重复。

81个difficult目标全部保留为普通目标。XML坐标视为连续xyxy边界，不额外加减1，逐框反算核对；这不是人工漏标/错标审计。与使用其他划分或VOC difficult忽略协议的论文成绩不直接比较。

数据来源报告、每张图片SHA256、最终划分见reports目录。完整数据不在本仓库分发，使用时遵守原始来源的条件。
