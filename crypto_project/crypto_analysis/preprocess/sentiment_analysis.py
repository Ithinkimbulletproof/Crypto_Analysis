import logging
import spacy
import torch
from django.db import transaction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from crypto_analysis.models import NewsArticle, SentimentData, KeyEntity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

vader_analyzer = SentimentIntensityAnalyzer()

bert_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).to(device)

emotion_analyzer = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,
    device=0 if device == "cuda" else -1,
)

topic_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if device == "cuda" else -1,
)

nlp = spacy.load("en_core_web_lg")

BATCH_SIZE = 256
CANDIDATE_TOPICS = [
    "DeFi",
    "NFT",
    "Regulation",
    "Exchange",
    "Mining",
    "Wallet",
    "DAO",
    "Stablecoin",
    "Security",
    "Partnership",
]


def analyze_batch(articles):
    try:
        texts = [prepare_text(article) for article in articles]

        with torch.no_grad():
            bert_inputs = bert_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            bert_outputs = bert_model(**bert_inputs)
            bert_probs = torch.softmax(bert_outputs.logits, dim=1).cpu().numpy()

        results = []
        for i, article in enumerate(articles):
            result = {
                "article": article,
                "text": texts[i],
                "vader": vader_analyzer.polarity_scores(texts[i]),
                "bert_positive": bert_probs[i][1],
                "bert_negative": bert_probs[i][0],
                "emotions": process_emotions(texts[i]),
                "topics": process_topics(texts[i]),
                "entities": process_entities(texts[i]),
            }
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return []


def prepare_text(article):
    return (
        f"{article.title}. {article.description}".encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )


def process_emotions(text):
    try:
        emotion_scores = emotion_analyzer(text[:512])[0]
        return {item["label"]: item["score"] for item in emotion_scores}
    except Exception as e:
        logger.error(f"Emotion analysis failed: {str(e)}")
        return {}


def process_topics(text):
    try:
        result = topic_classifier(
            text[:1024], candidate_labels=CANDIDATE_TOPICS, multi_label=True
        )
        return {
            label: score for label, score in zip(result["labels"], result["scores"])
        }
    except Exception as e:
        logger.error(f"Topic classification failed: {str(e)}")
        return {}


def process_entities(text):
    try:
        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "GPE", "MONEY"]:
                entities.setdefault(ent.label_, []).append(ent.text)
        return {k: list(set(v))[:5] for k, v in entities.items()}
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        return {}


@transaction.atomic
def save_results(batch_results):
    try:
        for result in batch_results:
            sentiment, created = SentimentData.objects.get_or_create(
                article=result["article"],
                defaults={
                    "vader_compound": result["vader"]["compound"],
                    "bert_positive": result["bert_positive"],
                    "emotion_scores": result["emotions"],
                    "topic_scores": result["topics"],
                    "combined_score": calculate_combined_score(result),
                },
            )

            if not created:
                continue

            for label, entities in result["entities"].items():
                for entity in entities:
                    KeyEntity.objects.get_or_create(
                        article=result["article"], entity_type=label, text=entity[:255]
                    )

    except Exception as e:
        logger.error(f"Save failed: {str(e)}")


def calculate_combined_score(result):
    emotion_weights = {"joy": 0.4, "optimism": 0.3, "anger": -0.5, "sadness": -0.4}

    emotion_score = sum(
        score * emotion_weights.get(label, 0)
        for label, score in result["emotions"].items()
    )

    return (
        (result["bert_positive"] * 0.6)
        + (result["vader"]["compound"] * 0.4)
        + emotion_score
    )


def analyze_sentiment():
    try:
        queryset = NewsArticle.objects.filter(sentiment_data__isnull=True)
        total = queryset.count()

        logger.info(f"Начинаем расширенный анализ для {total} статей")

        for i in range(0, total, BATCH_SIZE):
            batch = list(queryset[i : i + BATCH_SIZE])
            batch_results = analyze_batch(batch)
            save_results(batch_results)

            logger.info(f"Обработано {min(i + BATCH_SIZE, total)}/{total} статей")

            if device == "cuda":
                torch.cuda.empty_cache()

        logger.info("Анализ успешно завершен")
        return True

    except Exception as e:
        logger.error(f"Анализ не выполнен: {str(e)}")
        return False


if __name__ == "__main__":
    analyze_sentiment()
