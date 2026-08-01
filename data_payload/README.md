# data_payload

通过 git 中转到新集群的小文件（OSS 不方便直接上传时用）。

## llava_hr_essential_sa1b_ivcap.json.gz

SFT 数据 JSON（原始 292MB，gzip 后 53MB）。
原始文件 md5: `a612fd01237272a535c799f62aa33319`

在新集群上还原并放到 OSS：

```bash
cd ~/ml-fastvlm && git pull
gunzip -kc data_payload/llava_hr_essential_sa1b_ivcap.json.gz \
    > /data/oss_bucket_0/wangziyi/models_data/sft_data/llava_hr_essential_sa1b_ivcap.json
md5sum /data/oss_bucket_0/wangziyi/models_data/sft_data/llava_hr_essential_sa1b_ivcap.json
# 应为 a612fd01237272a535c799f62aa33319
```

该 JSON 引用的图片前缀目录（train_split 下需要能访问到）：
`ocr_vqa` `sa1b` `coco` `vg` `synthdog` `gqa` `docvqa` `infovqa`
