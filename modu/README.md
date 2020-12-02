# modu NER

- [모두의 말뭉치](https://corpus.korean.go.kr/) 개체명 분석 말뭉치 이용

  - `monologg/kocharelectra-base-modu-ner-sx`: **구어체** 사용하여 학습
  - `monologg/kocharelectra-base-modu-ner-nx`: **문어체** 사용하여 학습
  - `monologg/kocharelectra-base-modu-ner-all`: **문어체, 구어체** 모두 사용하여 학습

- Train:Test = 9:1 로 하여 진행 (macro F1 측정)
  - `sx`: 89.07%
  - `nx`: 90.74%
  - `all`: 90.73%

## Prerequisite

`transformers` 버전은 반드시 지켜주세요

- torch==1.7.0
- transformers==3.5.1

## How to use

- `grouped_entities=True`를 통해 `B`, `I` 태그 묶여짐
- device 지정 가능 (cpu: -1)
- `ignore_labels=[]`로 세팅하면 `O` 라벨이 달린 것도 모두 볼 수 있음

```python
from transformers import ElectraForTokenClassification, TokenClassificationPipeline
from tokenization_kocharelectra import KoCharElectraTokenizer
from pprint import pprint

tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-modu-ner-all")
model = ElectraForTokenClassification.from_pretrained("monologg/kocharelectra-base-modu-ner-all")

ner = TokenClassificationPipeline(
    model=model, tokenizer=tokenizer, ignore_labels=["O"], grouped_entities=True, device=-1
)


pprint(
    ner(
        "문재인 대통령은 28일 서울 코엑스에서 열린 ‘데뷰 (Deview) 2019’ 행사에 참석해 젊은 개발자들을 격려하면서 우리 정부의 인공지능 기본구상을 내놓았다. 출처 : 미디어오늘 (http://www.mediatoday.co.kr)"
    )
)

# Out
[{'entity_group': 'PS', 'score': 0.9998801747957865, 'word': '문재인'},
 {'entity_group': 'CV', 'score': 0.9998312791188558, 'word': '대통령'},
 {'entity_group': 'DT', 'score': 0.9998779296875, 'word': '28일'},
 {'entity_group': 'LC', 'score': 0.9993503987789154, 'word': '서울'},
 {'entity_group': 'AF', 'score': 0.9995182752609253, 'word': '코엑스'},
 {'entity_group': 'EV', 'score': 0.8551274935404459, 'word': '데뷰'},
 {'entity_group': 'EV', 'score': 0.7690083235502243, 'word': 'Deview'},
 {'entity_group': 'DT', 'score': 0.9989355504512787, 'word': '2019'},
 {'entity_group': 'CV', 'score': 0.9997546076774597, 'word': '개발자'},
 {'entity_group': 'OG', 'score': 0.9998011291027069, 'word': '정부'},
 {'entity_group': 'FD', 'score': 0.999186784029007, 'word': '인공지능'},
 {'entity_group': 'OG', 'score': 0.9976320862770081, 'word': '미디어오늘'},
 {'entity_group': 'TM',
  'score': 0.9994067351023356,
  'word': 'http://www.mediatoday.co.kr'}]
```

## Acknowledgement

해당 NER 모델은 **모두의 말뭉치** 도움으로 제작되었습니다.
