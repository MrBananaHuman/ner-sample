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