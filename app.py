import streamlit as st
from flair.data import Sentence
from flair.models import SequenceTagger

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn.functional as F



ckpt_path="amphora/FinABSA"
NER_tag_list = ['ORG']
@st.cache_resource
def load_model():
    return AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
ABSA = load_model()
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
tagger = SequenceTagger.load('ner')

def run_single_absa(input_str, tgt):
    input_str = input_str.replace(tgt, '[TGT]')
    input = tokenizer(input_str, return_tensors='pt')

    output = ABSA.generate(
        **input,
        max_length=20,
        output_scores=True,
        return_dict_in_generate=True
    )

    classification_output = tokenizer.convert_ids_to_tokens(
        int(output['sequences'][0][-4])
    )
    logits = F.softmax(output['scores'][-4][:, -3:], dim=1)[0]

    return {
        "classification_output": classification_output,
        "logits":
            {
                'positive': float(logits[0]),
                'negative': float(logits[1]),
                'neutral': float(logits[2])
            }
    }
def retrieve_target(input_str):
    sentence = Sentence(input_str)
    tagger.predict(sentence)
    entities = [entity.text for entity in sentence.get_spans('ner') if entity.tag in NER_tag_list]
    return entities
def run_absa(input_str):
    tgt_entities = retrieve_target(input_str)
    output = {e: run_single_absa(input_str, e) for e in tgt_entities}
    return output

user_input = st.text_area("Enter Financial News for Sentiment Analysis")
button = st.button("Analyze")

if user_input and button:
    final_output = run_absa(user_input)
    for Entity_Name in final_output:
        st.write("Company : ",Entity_Name,", Sentiment : ",final_output[Entity_Name]['classification_output'],", Score : ",final_output[Entity_Name]['logits'][final_output[Entity_Name]['classification_output'].lower()])