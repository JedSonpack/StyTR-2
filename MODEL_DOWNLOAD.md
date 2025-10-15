# 预训练模型下载

由于模型文件太大，无法直接上传到 GitHub，请从以下链接下载预训练模型：

## 下载链接

请从原作者提供的 Google Drive 下载以下预训练模型文件：

- **vgg-model**: [下载链接](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing)
- **vit_embedding**: [下载链接](https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing)
- **decoder**: [下载链接](https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing)
- **Transformer_module**: [下载链接](https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing)

## 安装说明

1. 下载所有模型文件
2. 将下载的文件放入 `experiments/` 文件夹
3. 确保文件名为：
   - `vgg_normalised.pth`
   - `embedding_iter_160000.pth`
   - `decoder_iter_160000.pth`
   - `transformer_iter_160000.pth`

## 使用方法

完成模型下载后，可以运行测试：

```bash
python test.py --content input/content/content_image.png --style input/style/style_image.png --output output
