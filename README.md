# Food Not Food Text Classifier

My first model built and shipped with Hugging Face. It's a text classifier that reads a short caption and decides whether it describes food or something else.

## What it does

Give it a sentence like "A bowl of ramen with soft-boiled eggs and scallions" and it labels it `food`. Give it "A yellow tractor driving over a hill" and it labels it `not_food`. Each prediction comes with a confidence score.

## How it's built

- Base model: `distilbert-base-uncased`
- Fine-tuned with `transformers.Trainer` on a labeled dataset of image captions ([`mrdbourke/learn_hf_food_not_food_image_captions`](https://huggingface.co/datasets/mrdbourke/learn_hf_food_not_food_image_captions))
- 250 examples total, split 200 train / 50 test
- 10 epochs, batch size 32, learning rate 2e-5
- Trained in Google Colab on a free T4 GPU

The dataset is tiny, so the model hits 100% accuracy on the test split fast. That's a sign of a small, clean dataset more than a sign the model is bulletproof, it hasn't seen much variety yet. A few hand-written test sentences in the notebook (idioms, sarcasm, mixed topics) show where it still guesses wrong.

## Try it

The trained model is on the Hugging Face Hub: [`JoseA1988/learn_hf_food_not_food_text_classifier-distilbert-base-uncased`](https://huggingface.co/JoseA1988/learn_hf_food_not_food_text_classifier-distilbert-base-uncased)

Quick way, using `pipeline`:

```python
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="JoseA1988/learn_hf_food_not_food_text_classifier-distilbert-base-uncased",
    top_k=1,
)

classifier("A delicious photo of a plate of scrambled eggs, toast and bacon")
# [[{'label': 'food', 'score': 0.97}]]
```

Manual way, with the tokenizer and model separately (useful if you want to see the raw logits or batch things yourself):

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "JoseA1988/learn_hf_food_not_food_text_classifier-distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

inputs = tokenizer("A yellow tractor driving over a hill", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

predicted_id = outputs.logits.argmax().item()
print(model.config.id2label[predicted_id])
```

## What's in this repo

- `huggingface_text_classification_tutorial_video.ipynb` — the full notebook: loading the dataset, tokenizing, training, evaluating, and pushing the model to the Hub.

## Notes on the workflow

Everything runs through the standard Hugging Face stack: `datasets` for loading and mapping the data, `AutoTokenizer` for turning text into token IDs, `AutoModelForSequenceClassification` for the classifier head on top of DistilBERT, and `Trainer` for the training loop and evaluation. The notebook also compares single-sample inference against batched inference with `pipeline`, batching cuts the average time per prediction noticeably once you're running more than a handful of samples at once.

## Next steps

- Grow the dataset past 250 examples, current size overfits fast
- Add trickier negative examples (food-adjacent objects, kitchen scenes without food itself)
- Try a base model bigger than DistilBERT and compare accuracy vs. speed
