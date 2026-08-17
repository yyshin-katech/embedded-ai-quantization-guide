# 성능 매트릭스 (자동 생성)

> 총 6 케이스 · TRT ['10.16.1.11']

## Latency (ms, median)

|                         |     fp16 |    fp32 |     int8 |
|:------------------------|---------:|--------:|---------:|
| ('resnet50', 'qcs8550') | nan      | nan     | nan      |
| ('resnet50', 'rtx')     |   1.0231 |   1.837 |   0.8628 |
| ('resnet50', 'rzv2h')   | nan      | nan     | nan      |
| ('resnet50', 'tda4vm')  | nan      | nan     | nan      |


## Accuracy (top-1)

|                         |     fp16 |     fp32 |    int8 |
|:------------------------|---------:|---------:|--------:|
| ('resnet50', 'qcs8550') | nan      | nan      | nan     |
| ('resnet50', 'rtx')     |   0.7686 |   0.7688 |   0.768 |
| ('resnet50', 'rzv2h')   | nan      | nan      | nan     |
| ('resnet50', 'tda4vm')  | nan      | nan      | nan     |

