"""
Topic Modeling Demo API
------------------------
Provides LDA and NMF topic modeling with word probabilities, sentence-level 
topic assignments, customizable vectorization parameters, and text statistics.
Supports PDF, DOCX, CSV, TSV, and plain text file uploads.
"""

import re
import csv
import io
import docx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from pypdf import PdfReader
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False

app = FastAPI(
    title="Topic Modeling Demo API",
    description="Advanced NLP & Topic Extraction Engine using LDA & NMF",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class TopicRequest(BaseModel):
    text: str
    num_topics: int = Field(default=3, ge=2, le=8)
    method: str = Field(default="lda")  # "lda" or "nmf"
    use_bigrams: bool = True
    max_words_per_topic: int = Field(default=8, ge=3, le=15)


class WordWeight(BaseModel):
    word: str
    weight: float


class Topic(BaseModel):
    id: int
    label: str           # Human-readable label inferred from top word
    top_words: List[WordWeight]
    percentage: float    # Percentage of text belonging to this topic


class SegmentAssignment(BaseModel):
    id: int
    text: str
    dominant_topic: int
    topic_label: str     # Matches Topic.label
    confidence: float


class TextStats(BaseModel):
    total_characters: int
    total_words: int
    total_sentences: int
    vocab_size: int


class TopicResponse(BaseModel):
    topics: List[Topic]
    segments: List[SegmentAssignment]
    stats: TextStats
    method_used: str
    num_segments_used: int


class SuggestTopicsResponse(BaseModel):
    suggested_num_topics: int
    reason: str


class UploadResponse(BaseModel):
    filename: str
    content_type: str
    extracted_text: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Preset Data
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "tech": {
        "title": "AI & Cloud Tech",
        "icon": "Cpu",
        "description": "Machine learning, neural networks, cloud computing, and cybersecurity.",
        "text": (
            "Artificial intelligence and machine learning models require large computational resources and high quality datasets for training. "
            "Deep neural networks perform representation learning on complex structured and unstructured data. "
            "Cloud infrastructure platforms like AWS, Google Cloud, and Azure allow software developers to deploy scalable AI microservices. "
            "Cybersecurity experts monitor global network traffic to protect enterprise cloud computing systems from security threats. "
            "Data engineering pipelines clean and aggregate telemetry before feeding data into predictive machine learning models. "
            "DevOps automation speeds up software delivery cycles across hybrid cloud server networks."
        )
    },
    "economics": {
        "title": "Global Finance & Economy",
        "icon": "TrendingUp",
        "description": "Central banking, stock markets, inflation, and startup investments.",
        "text": (
            "Central banks adjusted benchmark interest rates today to control rising inflation and stabilize currency valuations. "
            "Stock market indices experienced significant volatility as quarterly corporate earnings reports were released by financial institutions. "
            "International trade agreements and supply chain logistics continue to influence global commodity and consumer prices. "
            "Venture capital funds and angel investors are backing early stage technology startups despite higher borrowing costs. "
            "Macroeconomic policy balances employment targets against interest rate adjustments to foster long term monetary growth. "
            "Investment portfolios are diversifying across equity assets, treasury bonds, and emerging market funds."
        )
    },
    "climate": {
        "title": "Climate & Renewable Energy",
        "icon": "Leaf",
        "description": "Solar power, carbon emissions, marine biology, and grid energy storage.",
        "text": (
            "Renewable energy generation from commercial solar farms and offshore wind turbines reached unprecedented capacity levels this year. "
            "Climate scientists emphasize urgent reductions in carbon emissions to mitigate extreme weather phenomena worldwide. "
            "Oceanic temperatures are rising at historical rates, threatening delicate marine biodiversity and coastal coral reef ecosystems. "
            "Advanced grid battery storage technology is critical for balancing renewable power distribution across national electric grids. "
            "Environmental policy frameworks mandate sustainable land management and clean energy transitions for heavy industry. "
            "Conservation initiatives focus on preserving rainforest habitats and protecting endangered wildlife species."
        )
    },
    "health": {
        "title": "Healthcare & Genomics",
        "icon": "Activity",
        "description": "Immunotherapy, genomic sequencing, medical imaging, and drug discovery.",
        "text": (
            "Medical researchers discovered a novel targeted immunotherapy treatment that selectively targets aggressive cancer cell receptors. "
            "Clinical trial phases demonstrated significant therapeutic efficacy and minimal adverse side effects compared to traditional chemotherapy. "
            "Next-generation genomic sequencing enables precision medicine tailored to individual patient genetic profiles and biomarkers. "
            "Diagnostic imaging software powered by computer vision assists radiologists in early detection of internal organ pathologies. "
            "Pharmaceutical laboratories are accelerating drug discovery using high-throughput molecular screening algorithms. "
            "Public health organizations monitor disease epidemiology to prevent widespread seasonal infectious outbreaks."
        )
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_segments(text: str, target_num_segments: int = 3) -> List[str]:
    """Break text into sentence, paragraph, or word-window chunks."""
    raw_chunks = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    segments = [s.strip() for s in raw_chunks if len(s.strip()) >= 8]

    if len(segments) < target_num_segments:
        words = re.findall(r"\b\w+\b", text.strip())
        if len(words) >= target_num_segments * 4:
            chunk_size = max(8, len(words) // target_num_segments)
            segments = [
                " ".join(words[i : i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]
            segments = [s for s in segments if len(s) >= 8]

    return segments


def infer_topic_label(top_word: str) -> str:
    """Return a capitalized human-readable label from the top word of a topic."""
    # Strip common suffixes to get a cleaner root word
    word = top_word.split()[0]  # Use only first word if bigram
    return word.capitalize()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok", "version": "2.1.0"}


@app.get("/presets")
def get_presets():
    return PRESETS


@app.get("/suggest-topics", response_model=SuggestTopicsResponse)
def suggest_topics(text: str = Query(..., min_length=20)):
    """
    Heuristically suggest a good number of topics based on the text's
    vocabulary diversity and sentence count.
    """
    cleaned = text.strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if len(s.strip()) >= 8]
    words = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower())

    # Very rough English stopword set for diversity scoring
    STOP = {
        "the","and","for","that","this","are","was","were","has","have","had",
        "its","with","from","they","their","which","will","can","but","not",
        "all","one","our","more","also","each","been","into","than","then",
        "some","would","about","after","before","when","where","while","how"
    }
    content_words = [w for w in words if w not in STOP]
    unique_content = len(set(content_words))
    total_sentences = len(sentences)

    # Scoring logic
    if total_sentences <= 3 or unique_content < 20:
        suggested = 2
        reason = "Short text with limited vocabulary — 2 topics recommended."
    elif total_sentences <= 6 or unique_content < 50:
        suggested = 3
        reason = f"Moderate text ({total_sentences} sentences, {unique_content} unique content words) — 3 topics recommended."
    elif total_sentences <= 10 or unique_content < 100:
        suggested = 4
        reason = f"Rich text ({total_sentences} sentences, {unique_content} unique content words) — 4 topics recommended."
    elif unique_content < 180:
        suggested = 5
        reason = f"Dense text ({unique_content} unique content words) — 5 topics recommended."
    else:
        suggested = 6
        reason = f"Very large, diverse text ({unique_content} unique content words, {total_sentences} sentences) — 6 topics recommended."

    return SuggestTopicsResponse(suggested_num_topics=suggested, reason=reason)


@app.post("/predict", response_model=TopicResponse)
def predict_topics(req: TopicRequest):
    cleaned_text = req.text.strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    segments = split_into_segments(cleaned_text, target_num_segments=req.num_topics)

    if len(segments) < req.num_topics:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text is too short to extract {req.num_topics} distinct topics. "
                f"Please paste a larger body of text containing at least {req.num_topics} sentences or paragraphs."
            ),
        )

    ngram_range = (1, 2) if req.use_bigrams else (1, 1)

    try:
        if req.method == "nmf":
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=ngram_range,
                max_df=0.95,
                min_df=1,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
            )
            matrix = vectorizer.fit_transform(segments)

            n_features = matrix.shape[1]
            if n_features == 0:
                raise ValueError("No valid words found after removing stop words.")
            if n_features < req.num_topics:
                raise ValueError(
                    f"Text vocabulary has only {n_features} unique words, but {req.num_topics} topics were requested."
                )

            model = NMF(
                n_components=req.num_topics,
                random_state=42,
                init="random",
                max_iter=500
            )
            # NMF can fail with sparse matrices on some numpy builds; use dense
            matrix = np.ascontiguousarray(matrix.toarray())
        else:
            vectorizer = CountVectorizer(
                stop_words="english",
                ngram_range=ngram_range,
                max_df=0.95,
                min_df=1,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
            )
            matrix = vectorizer.fit_transform(segments)

            n_features = matrix.shape[1]
            if n_features == 0:
                raise ValueError("No valid words found after removing stop words.")

            model = LatentDirichletAllocation(
                n_components=req.num_topics,
                random_state=42,
                max_iter=50
            )

        doc_topic_matrix = model.fit_transform(matrix)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Topic extraction error: {str(e)}. Try pasting a larger or more diverse text body."
        )

    feature_names = vectorizer.get_feature_names_out()

    # Build topics with inferred labels
    dominant_counts = np.zeros(req.num_topics)
    topics: List[Topic] = []
    topic_labels: List[str] = []

    for topic_idx, component in enumerate(model.components_):
        comp_sum = component.sum()
        normalized_weights = component / comp_sum if comp_sum > 0 else component

        top_indices = component.argsort()[-req.max_words_per_topic:][::-1]
        word_weights: List[WordWeight] = []

        for i in top_indices:
            w = feature_names[i]
            score = float(normalized_weights[i])
            word_weights.append(WordWeight(word=w, weight=round(score, 4)))

        label = infer_topic_label(feature_names[top_indices[0]]) if len(top_indices) > 0 else f"Topic {topic_idx + 1}"
        topic_labels.append(label)
        # Placeholder percentage — filled in after segment pass
        topics.append(
            Topic(
                id=topic_idx,
                label=label,
                top_words=word_weights,
                percentage=0.0
            )
        )

    # Assign segments to topics
    segment_assignments: List[SegmentAssignment] = []
    for idx, seg in enumerate(segments):
        dist = doc_topic_matrix[idx]
        dom_topic = int(np.argmax(dist))
        dist_sum = np.sum(dist)
        confidence = float(dist[dom_topic] / dist_sum) if dist_sum > 0 else 1.0 / req.num_topics
        dominant_counts[dom_topic] += 1
        segment_assignments.append(
            SegmentAssignment(
                id=idx,
                text=seg,
                dominant_topic=dom_topic,
                topic_label=topic_labels[dom_topic],
                confidence=round(confidence, 3)
            )
        )

    # Fill in percentages
    total_segs = max(len(segments), 1)
    topic_percentages = (dominant_counts / total_segs) * 100
    for t in topics:
        t.percentage = round(float(topic_percentages[t.id]), 1)

    # Text statistics
    words = re.findall(r"\b\w+\b", cleaned_text)
    stats = TextStats(
        total_characters=len(cleaned_text),
        total_words=len(words),
        total_sentences=len(segments),
        vocab_size=len(feature_names)
    )

    return TopicResponse(
        topics=topics,
        segments=segment_assignments,
        stats=stats,
        method_used=req.method.upper(),
        num_segments_used=len(segments)
    )


@app.post("/export-csv")
def export_csv(results: TopicResponse):
    """
    Accept a TopicResponse payload and stream it back as a downloadable CSV.
    The CSV has two sheets merged: topic summary rows followed by segment rows.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # --- Topic Summary Section ---
    writer.writerow(["=== TOPIC SUMMARY ==="])
    writer.writerow(["Topic ID", "Label", "Percentage (%)", "Top Words (weight)"])
    for t in results.topics:
        top_words_str = " | ".join(
            [f"{w.word} ({w.weight:.3f})" for w in t.top_words]
        )
        writer.writerow([f"Topic {t.id + 1}", t.label, t.percentage, top_words_str])

    writer.writerow([])  # blank separator

    # --- Segment Section ---
    writer.writerow(["=== SEGMENT ASSIGNMENTS ==="])
    writer.writerow(["Segment #", "Dominant Topic", "Topic Label", "Confidence (%)", "Text"])
    for seg in results.segments:
        writer.writerow([
            seg.id + 1,
            f"Topic {seg.dominant_topic + 1}",
            seg.topic_label,
            f"{seg.confidence * 100:.1f}",
            seg.text
        ])

    writer.writerow([])
    writer.writerow(["Method", results.method_used])
    writer.writerow(["Total Words", results.stats.total_words])
    writer.writerow(["Vocabulary Size", results.stats.vocab_size])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=topic-results-{results.method_used.lower()}.csv"
        }
    )


@app.post("/upload-file", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    extracted_text = ""

    try:
        if filename.endswith(".pdf"):
            if not PDF_SUPPORTED:
                raise HTTPException(
                    status_code=501,
                    detail="PDF support is not available. Please install pypdf."
                )
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(page_text.strip())
            extracted_text = "\n\n".join(pages_text)
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            extracted_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        elif filename.endswith(".csv") or filename.endswith(".tsv"):
            delimiter = "\t" if filename.endswith(".tsv") else ","
            decoded_content = content.decode("utf-8-sig", errors="replace")
            reader = csv.reader(io.StringIO(decoded_content), delimiter=delimiter)
            rows_text = []
            for row in reader:
                clean_row = " ".join([cell.strip() for cell in row if cell.strip()])
                if clean_row:
                    rows_text.append(clean_row)
            extracted_text = "\n".join(rows_text)
        else:
            # Fallback for txt, md, json, log, etc.
            extracted_text = content.decode("utf-8", errors="replace")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from file: {str(e)}")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded file.")

    return UploadResponse(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        extracted_text=extracted_text.strip(),
        size_bytes=len(content)
    )
