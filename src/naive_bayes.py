import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Load dataset
data = pd.read_csv("data/corpus_small.csv")

texts = data["text"].values
labels = data["label"].values

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

processed_texts = [preprocess(text) for text in texts]

class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocabulary = set()
        self.class_priors = {}
        self.word_probabilities = {}

    def fit(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for word in text:
                self.class_word_counts[label][word] += 1
                self.vocabulary.add(word)

        total_docs = len(labels)

        for label in self.class_counts:
            self.class_priors[label] = self.class_counts[label] / total_docs

        for label in self.class_word_counts:
            self.word_probabilities[label] = {}
            total_words = sum(self.class_word_counts[label].values())

            for word in self.vocabulary:
                self.word_probabilities[label][word] = (
                    (self.class_word_counts[label][word] + 1) /
                    (total_words + len(self.vocabulary))
                )

    def predict(self, text):
        words = preprocess(text)
        scores = {}

        for label in self.class_priors:
            score = np.log(self.class_priors[label])
            for word in words:
                if word in self.vocabulary:
                    score += np.log(self.word_probabilities[label][word])
            scores[label] = score

        return max(scores, key=scores.get)

model = NaiveBayesClassifier()
model.fit(processed_texts, labels)

test_sentences = [
    "win money now",
    "call me later",
    "free cash prize",
    "see you tomorrow"
]

for sentence in test_sentences:
    print(sentence, "->", model.predict(sentence))
