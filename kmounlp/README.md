# kmounnlp NER

[한국해양대 NER](https://github.com/kmounlp/NER)

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

tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-kmounlp-ner")
model = ElectraForTokenClassification.from_pretrained("monologg/kocharelectra-base-kmounlp-ner")

ner = TokenClassificationPipeline(
    model=model, tokenizer=tokenizer, ignore_labels=["O"], grouped_entities=True, device=-1
)


pprint(
    ner(
        "문재인 대통령은 28일 서울 코엑스에서 열린 ‘데뷰 (Deview) 2019’ 행사에 참석해 젊은 개발자들을 격려하면서 우리 정부의 인공지능 기본구상을 내놓았다. 출처 : 미디어오늘 (http://www.mediatoday.co.kr)"
    )
)

# Out
[{'entity_group': 'PER', 'score': 0.9999483029047648, 'word': '문재인'},
 {'entity_group': 'DAT', 'score': 0.9998882214228312, 'word': '28일'},
 {'entity_group': 'LOC', 'score': 0.9997751414775848, 'word': '서울'},
 {'entity_group': 'LOC', 'score': 0.9619666735331217, 'word': '코엑스'},
 {'entity_group': 'POH', 'score': 0.9991543889045715, 'word': '데뷰'},
 {'entity_group': 'POH', 'score': 0.9901127517223358, 'word': 'Deview'},
 {'entity_group': 'DAT', 'score': 0.977840855717659, 'word': '2019'},
 {'entity_group': 'POH',
  'score': 0.999930754855827,
  'word': 'http://www.mediatoday.co.kr'}]
```
