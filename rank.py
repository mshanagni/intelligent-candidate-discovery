"""
Intelligent Candidate Discovery & Ranking System
Redrob Hackathon — Senior AI Engineer JD

Architecture:
  Multi-signal scoring engine that goes beyond keyword matching to understand
  semantic career fit, role legitimacy, behavioral engagement, and logistical fit.

Scoring Dimensions:
  1. Role Legitimacy     (20%) - Is this actually a technical AI/ML candidate?
  2. Technical Skills    (30%) - Do they have the RIGHT stack for retrieval/ranking?
  3. Career Quality      (20%) - Product company exp, right years, production deployment
  4. Semantic Fit        (15%) - TF-IDF cosine sim of career narratives vs JD
  5. Behavioral Signals  (10%) - Availability, response rate, engagement
  6. Location/Logistics  ( 5%) - India-preferred, notice period, relocation

Red Flag Multipliers (applied after weighted sum):
  - Non-technical role (HR/Marketing/Accountant/etc.)  → 0.04x
  - Entire career at consulting giants                  → 0.35x
  - Outside India + unwilling to relocate              → 0.20x
  - Honeypot (impossible timeline)                     → 0.0x

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — Job Description Signals
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer founding team Series A AI-native talent intelligence platform
Redrob AI Pune Noida India Hybrid.
Production experience embeddings retrieval systems sentence-transformers BGE E5 OpenAI embeddings
deployed real users embedding drift index refresh retrieval quality regression production.
Production experience vector databases hybrid search Pinecone Weaviate Qdrant Milvus FAISS
OpenSearch Elasticsearch operational experience ranking retrieval matching.
Strong Python code quality hands-on evaluation frameworks ranking systems NDCG MRR MAP
offline online A/B test interpretation.
LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural.
HR tech recruiting marketplace products distributed systems large-scale inference.
Open-source contributions AI ML space.
Shipped ranking search recommendation system production real users scale.
Hybrid retrieval dense retrieval evaluation offline online.
5 to 9 years experience applied ML AI product companies not pure services.
"""

# Required skills: group → (keywords, max_weight)
CORE_SKILLS = {
    "embeddings": {
        "terms": {
            "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5",
            "openai embeddings", "embedding", "dense retrieval", "bi-encoder", "cross-encoder",
            "semantic search", "semantic similarity", "text embeddings"
        },
        "weight": 2.5
    },
    "vector_db": {
        "terms": {
            "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
            "elasticsearch", "annoy", "scann", "pgvector", "vector database",
            "vector store", "vector index", "ann", "approximate nearest neighbor",
            "hybrid search", "knn search"
        },
        "weight": 2.5
    },
    "ranking_retrieval": {
        "terms": {
            "ranking", "retrieval", "information retrieval", "ir", "bm25", "tfidf",
            "learning to rank", "ltr", "reranking", "re-ranking", "ndcg", "mrr", "map",
            "recommendation", "recommender", "search engine", "search system",
            "candidate retrieval", "two-stage", "two stage", "recall", "precision"
        },
        "weight": 2.0
    },
    "python_ml": {
        "terms": {
            "python", "pytorch", "tensorflow", "sklearn", "scikit-learn", "numpy",
            "pandas", "transformers", "hugging face", "huggingface"
        },
        "weight": 1.5
    },
    "llm_finetuning": {
        "terms": {
            "fine-tuning", "finetuning", "fine tuning", "lora", "qlora", "peft",
            "llm", "large language model", "instruction tuning", "rlhf",
            "language model", "gpt", "bert", "llama", "mistral"
        },
        "weight": 1.2
    },
    "mlops_eval": {
        "terms": {
            "mlflow", "wandb", "weights & biases", "evaluation", "benchmarking",
            "a/b test", "a/b testing", "experiment tracking", "model serving",
            "model deployment", "inference", "triton", "torchserve", "bentoml"
        },
        "weight": 1.0
    },
    "production_signals": {
        "terms": {
            "production", "deployed", "shipped", "at scale", "millions", "billions",
            "latency", "throughput", "real users", "live system", "online system",
            "production traffic", "serving", "microservices", "api"
        },
        "weight": 1.3
    },
}

SKILL_NAME_MAP = {
    # canonical name → group
    "embeddings": "embeddings", "sentence-transformers": "embeddings",
    "sentence transformers": "embeddings", "bge": "embeddings", "e5": "embeddings",
    "semantic search": "embeddings", "text embeddings": "embeddings",
    "pinecone": "vector_db", "weaviate": "vector_db", "qdrant": "vector_db",
    "milvus": "vector_db", "faiss": "vector_db", "opensearch": "vector_db",
    "elasticsearch": "vector_db", "pgvector": "vector_db", "annoy": "vector_db",
    "ranking": "ranking_retrieval", "retrieval": "ranking_retrieval",
    "information retrieval": "ranking_retrieval", "bm25": "ranking_retrieval",
    "recommendation systems": "ranking_retrieval", "recommender": "ranking_retrieval",
    "ltr": "ranking_retrieval", "learning to rank": "ranking_retrieval",
    "ndcg": "ranking_retrieval", "reranking": "ranking_retrieval",
    "python": "python_ml", "pytorch": "python_ml", "tensorflow": "python_ml",
    "scikit-learn": "python_ml", "sklearn": "python_ml",
    "hugging face": "python_ml", "huggingface": "python_ml",
    "hugging face transformers": "python_ml", "transformers": "python_ml",
    "fine-tuning llms": "llm_finetuning", "lora": "llm_finetuning",
    "qlora": "llm_finetuning", "peft": "llm_finetuning",
    "mlflow": "mlops_eval", "weights & biases": "mlops_eval", "wandb": "mlops_eval",
    "haystack": "ranking_retrieval", "langchain": "llm_finetuning",
    "llamaindex": "ranking_retrieval", "llama index": "ranking_retrieval",
    "xgboost": "ranking_retrieval", "lightgbm": "ranking_retrieval",
}

PROFICIENCY_WEIGHT = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.60, "beginner": 0.30}

# ---------------------------------------------------------------------------
# Role Classification
# ---------------------------------------------------------------------------

# Titles that should score very low for this JD
NON_TECHNICAL_TITLES = {
    "hr manager", "human resources", "marketing manager", "content writer",
    "accountant", "graphic designer", "customer support", "operations manager",
    "sales", "business analyst", "project manager", "civil engineer",
    "mechanical engineer", "electrical engineer", "product manager",
    "scrum master", "finance", "recruiter", "procurement", "supply chain",
    "logistics", "legal", "compliance", "administrator", "admin",
    "java developer", ".net developer", "sap consultant", "oracle",
    "mobile developer", "android developer", "ios developer",
    "ui designer", "ux designer", "frontend engineer",
    "full stack developer", "web developer",
}

# Strong positive title signals
STRONG_AI_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer", "research scientist",
    "nlp engineer", "data scientist", "applied scientist", "research engineer",
    "recommendation systems engineer", "search engineer", "retrieval engineer",
    "ranking engineer", "ai/ml engineer", "computer vision engineer",
    "deep learning engineer", "llm engineer", "senior ml", "staff ml",
    "principal ml", "senior ai", "staff ai",
}

MODERATE_TECH_TITLES = {
    "software engineer", "backend engineer", "data engineer", "cloud engineer",
    "platform engineer", "infrastructure engineer", "devops engineer", "sre",
    "solutions architect", "tech lead", "senior engineer",
}

# Consulting giants that are red flags if entire career is there
CONSULTING_GIANTS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "l&t infotech",
    "ltimindtree", "hexaware", "niit", "kpit", "persistent systems",
    "cyient", "zensar", "birlasoft"
}

# India-preferred locations
INDIA_TIER1_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram", "chennai", "kolkata", "ahmedabad",
    "noida", "greater noida",
}

# ---------------------------------------------------------------------------
# Honeypot Detection
# ---------------------------------------------------------------------------

# Known fictional companies in dataset (Dunder Mifflin = The Office, etc.)
FICTIONAL_COMPANIES = {"dunder mifflin", "initech", "globex inc", "globex",
                       "pied piper", "hooli", "wayne enterprises", "stark industries",
                       "acme corp", "umbrella corp"}

def detect_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with impossible profiles:
    - Experience claimed longer than time since graduation
    - Working at company before it was founded (simplified: just check timeline sanity)
    - Total experience claimed vs career_history sum mismatch
    """
    profile = candidate["profile"]
    career = candidate["career_history"]
    yoe = profile.get("years_of_experience", 0)

    today = date.today()

    # Check education timeline vs experience
    edu = candidate.get("education", [])
    if edu:
        earliest_grad = min(
            (e.get("end_year", 9999) for e in edu if e.get("end_year")),
            default=None
        )
        if earliest_grad and earliest_grad < 1960:
            return True  # impossible grad year
        if earliest_grad and earliest_grad > today.year:
            return True  # future grad = impossible
        if earliest_grad:
            max_possible_yoe = today.year - earliest_grad
            if yoe > max_possible_yoe + 2:  # allow 2yr buffer
                return True

    # Check career history timeline
    for job in career:
        start = job.get("start_date", "")
        if start:
            try:
                start_dt = datetime.strptime(start[:10], "%Y-%m-%d").date()
                if start_dt > today:
                    return True  # future start = honeypot
                if start_dt.year < 1970:
                    return True  # impossibly old
            except ValueError:
                pass

        # Check impossible duration at company
        end = job.get("end_date", "")
        if start and end:
            try:
                s = datetime.strptime(start[:10], "%Y-%m-%d").date()
                e = datetime.strptime(end[:10], "%Y-%m-%d").date()
                months = (e.year - s.year) * 12 + (e.month - s.month)
                stated = job.get("duration_months", 0)
                if abs(months - stated) > 24:  # >2yr discrepancy
                    return True
            except ValueError:
                pass

    # Check total career duration vs claimed experience
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if total_career_months > (yoe + 5) * 12:  # >5yr surplus
        return True
    if yoe > 3 and total_career_months < yoe * 12 * 0.3:  # <30% accounted
        return True

    return False


# ---------------------------------------------------------------------------
# Skill Scoring
# ---------------------------------------------------------------------------

def score_skills(candidate: dict) -> tuple[float, set[str]]:
    """
    Score based on skills list. Returns (score 0-1, matched_groups).
    """
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})

    group_scores: dict[str, float] = {}
    matched_groups: set[str] = set()

    for skill in skills:
        name = skill["name"].lower().strip()
        proficiency = skill.get("proficiency", "intermediate")
        endorsements = skill.get("endorsements", 0)
        duration_months = skill.get("duration_months", 0)

        # Resolve to group
        group = SKILL_NAME_MAP.get(name)
        if group is None:
            # Fuzzy match: check if skill name contains any core term
            for grp, data in CORE_SKILLS.items():
                for term in data["terms"]:
                    if term in name or name in term:
                        group = grp
                        break
                if group:
                    break

        if group is None:
            continue

        matched_groups.add(group)
        max_w = CORE_SKILLS[group]["weight"]

        # Proficiency base
        prof_w = PROFICIENCY_WEIGHT.get(proficiency, 0.5)

        # Endorsement bonus (log scale, capped at 0.3 bonus)
        endorse_bonus = min(0.3, math.log1p(endorsements) / 15)

        # Duration bonus (capped at 0.2)
        duration_bonus = min(0.2, duration_months / 60)

        # Assessment score bonus (if available)
        assess_bonus = 0.0
        skill_name_orig = skill["name"]
        if skill_name_orig in assessment_scores:
            assess_bonus = (assessment_scores[skill_name_orig] / 100) * 0.2

        raw = prof_w * (1 + endorse_bonus + duration_bonus + assess_bonus)
        group_scores[group] = max(group_scores.get(group, 0), min(1.0, raw))

    # Weighted sum across groups, normalized to [0, 1]
    total_weight = sum(CORE_SKILLS[g]["weight"] for g in CORE_SKILLS)
    weighted_sum = sum(
        group_scores.get(g, 0) * CORE_SKILLS[g]["weight"] for g in CORE_SKILLS
    )
    skill_score = weighted_sum / total_weight

    return skill_score, matched_groups


# ---------------------------------------------------------------------------
# Role Legitimacy Score
# ---------------------------------------------------------------------------

def score_role_legitimacy(candidate: dict) -> tuple[float, str]:
    """
    Returns (score 0-1, role_tier).
    role_tier: 'strong_ai', 'moderate_tech', 'adjacent', 'non_technical'
    """
    profile = candidate["profile"]
    title = profile.get("current_title", "").lower()
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()

    # Check career history for AI/ML roles
    career = candidate.get("career_history", [])
    career_titles = [j.get("title", "").lower() for j in career]
    all_titles = [title] + career_titles

    # Strong AI match
    for t in all_titles:
        for ai_t in STRONG_AI_TITLES:
            if ai_t in t:
                return 1.0, "strong_ai"

    # Check summary/headline for AI keywords
    ai_keywords = {"machine learning", "ml engineer", "ai engineer", "nlp", "deep learning",
                   "neural network", "retrieval", "ranking", "recommendation", "embedding",
                   "llm", "language model", "vector", "search engine"}
    ai_count = sum(1 for k in ai_keywords if k in summary or k in headline)

    # Moderate tech
    for t in all_titles:
        for mod_t in MODERATE_TECH_TITLES:
            if mod_t in t:
                if ai_count >= 2:
                    return 0.75, "moderate_tech_ai"
                return 0.55, "moderate_tech"

    # Non-technical
    for t in all_titles:
        for bad_t in NON_TECHNICAL_TITLES:
            if bad_t in t:
                # Even if they list AI skills, wrong role
                if ai_count >= 3:
                    return 0.25, "adjacent"
                return 0.08, "non_technical"

    # Unclear — use AI keyword count
    if ai_count >= 4:
        return 0.65, "moderate_tech_ai"
    elif ai_count >= 2:
        return 0.45, "adjacent"
    return 0.2, "non_technical"


# ---------------------------------------------------------------------------
# Career Quality Score
# ---------------------------------------------------------------------------

def score_career_quality(candidate: dict) -> float:
    """
    Score career depth and quality signals:
    - Years of experience (sweet spot 6-8)
    - Product company vs pure consulting
    - Production deployment evidence in descriptions
    - AI/ML career progression
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)

    # 1. Experience years score (6-8 ideal per JD)
    if 6 <= yoe <= 8:
        yoe_score = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        yoe_score = 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        yoe_score = 0.65
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        yoe_score = 0.45
    else:
        yoe_score = 0.25

    # 2. Company quality: product company vs pure consulting
    consulting_months = 0
    product_months = 0
    total_months = sum(j.get("duration_months", 0) for j in career)

    for job in career:
        company = job.get("company", "").lower().strip()
        months = job.get("duration_months", 0)
        industry = job.get("industry", "").lower()
        size = job.get("company_size", "")

        is_consulting = any(cg in company for cg in CONSULTING_GIANTS)
        is_it_services = "it services" in industry or "consulting" in industry

        if is_consulting or (is_it_services and not is_consulting):
            consulting_months += months
        else:
            product_months += months

    if total_months > 0:
        consulting_ratio = consulting_months / total_months
    else:
        consulting_ratio = 0.5

    # Product company experience valued
    company_score = 1.0 - (consulting_ratio * 0.7)  # 100% consulting → 0.3

    # 3. Production deployment evidence in career descriptions
    prod_keywords = {
        "production", "deployed", "shipped", "at scale", "real users",
        "live", "online", "latency", "throughput", "ranking", "retrieval",
        "search", "recommendation", "embedding", "vector", "index",
    }
    career_text = " ".join(j.get("description", "") for j in career).lower()

    prod_count = sum(1 for kw in prod_keywords if kw in career_text)
    prod_score = min(1.0, prod_count / 8)

    # 4. AI/ML role progression — any prior AI/ML titles?
    ai_ml_months = 0
    for job in career:
        t = job.get("title", "").lower()
        is_ai = any(ai_t in t for ai_t in STRONG_AI_TITLES) or \
                any(mod in t for mod in {"data scientist", "ml", "ai", "nlp",
                                         "recommendation", "search"})
        if is_ai:
            ai_ml_months += job.get("duration_months", 0)

    ai_progression = min(1.0, ai_ml_months / max(total_months, 1) * 2)  # 50%+ = 1.0

    # 5. Research-only penalty (academic labs, pure researcher)
    researcher_terms = {"research scientist", "research engineer", "phd student",
                        "research intern", "postdoc", "professor", "academic"}
    is_pure_researcher = all(
        any(rt in j.get("title", "").lower() for rt in researcher_terms)
        or any(rt in j.get("industry", "").lower() for rt in {"academia", "university", "research"})
        for j in career
    )
    research_penalty = 0.5 if is_pure_researcher else 1.0

    career_score = (
        0.30 * yoe_score +
        0.25 * company_score +
        0.25 * prod_score +
        0.20 * ai_progression
    ) * research_penalty

    return min(1.0, career_score)


# ---------------------------------------------------------------------------
# Semantic Fit Score (TF-IDF style)
# ---------------------------------------------------------------------------

JD_KEY_BIGRAMS = {
    "embedding retrieval", "vector database", "hybrid search",
    "dense retrieval", "ranking system", "recommendation system",
    "search engine", "production deployment", "information retrieval",
    "nlp", "natural language processing", "evaluation framework",
    "offline evaluation", "online evaluation", "a/b testing",
    "large scale", "at scale", "real users", "fine tuning",
    "learning to rank", "semantic search", "candidate ranking",
    "job matching", "talent intelligence"
}

JD_KEY_UNIGRAMS = {
    "embeddings", "retrieval", "ranking", "recommendation", "search",
    "vector", "index", "pinecone", "faiss", "milvus", "weaviate", "qdrant",
    "elasticsearch", "opensearch", "bm25", "ndcg", "mrr", "map",
    "python", "pytorch", "transformers", "bert", "llm", "fine-tuning",
    "lora", "peft", "production", "deployed", "shipped", "scale",
    "encoder", "decoder", "attention", "semantic", "dense", "sparse",
    "haystack", "llamaindex", "langchain", "mlflow", "wandb",
    "xgboost", "lightgbm", "gradient boosting",
}

NEGATIVE_TERMS = {
    "computer vision", "image classification", "object detection", "yolo",
    "speech recognition", "text to speech", "tts", "robotics",
    "android", "ios", "mobile", "react native", "flutter",
    "accounting", "tally", "sap", "salesforce", "crm",
    "marketing", "seo", "content writing", "photoshop", "illustrator",
    "civil", "mechanical", "electrical",
}

def score_semantic_fit(candidate: dict) -> float:
    """
    Score how well career narratives match the JD using keyword overlap.
    TF-IDF approximated via term presence + frequency in career descriptions.
    """
    career = candidate.get("career_history", [])
    profile = candidate["profile"]
    skills = candidate.get("skills", [])

    # Build candidate text corpus
    texts = [
        profile.get("summary", ""),
        profile.get("headline", ""),
        " ".join(j.get("description", "") for j in career),
        " ".join(s["name"] for s in skills),
    ]
    full_text = " ".join(texts).lower()

    # Positive signal: JD unigrams found in candidate text
    pos_count = sum(1 for term in JD_KEY_UNIGRAMS if term in full_text)
    pos_score = min(1.0, pos_count / 12)  # 12+ = full score

    # Bigram bonus
    bigram_count = sum(1 for bg in JD_KEY_BIGRAMS if bg in full_text)
    bigram_bonus = min(0.3, bigram_count / 8)

    # Negative signal: irrelevant domain terms
    neg_count = sum(1 for term in NEGATIVE_TERMS if term in full_text)
    neg_penalty = min(0.4, neg_count * 0.08)

    semantic_score = max(0.0, (pos_score + bigram_bonus) - neg_penalty)
    return min(1.0, semantic_score)


# ---------------------------------------------------------------------------
# Behavioral Signals Score
# ---------------------------------------------------------------------------

def score_behavioral(candidate: dict) -> float:
    """
    Score based on Redrob behavioral signals:
    - Active candidate (recently active, open to work)
    - Responsive (response rate, response time)
    - Market validated (saved by recruiters, search appearances)
    - Interview reliable (completion rate)
    - Profile authenticity
    """
    sig = candidate.get("redrob_signals", {})
    today = date.today()

    # 1. Activity recency
    last_active = sig.get("last_active_date", "")
    if last_active:
        try:
            active_dt = datetime.strptime(last_active[:10], "%Y-%m-%d").date()
            days_inactive = (today - active_dt).days
            if days_inactive <= 30:
                recency_score = 1.0
            elif days_inactive <= 90:
                recency_score = 0.8
            elif days_inactive <= 180:
                recency_score = 0.5
            elif days_inactive <= 365:
                recency_score = 0.3
            else:
                recency_score = 0.1
        except ValueError:
            recency_score = 0.5
    else:
        recency_score = 0.3

    # 2. Open-to-work flag (strong signal)
    open_flag = 1.0 if sig.get("open_to_work_flag", False) else 0.4

    # 3. Recruiter response rate
    response_rate = sig.get("recruiter_response_rate", 0.5)
    response_score = response_rate  # already 0-1

    # 4. Response time (lower = better; cap at 168 hours = 1 week)
    avg_response_h = sig.get("avg_response_time_hours", 48)
    if avg_response_h <= 4:
        rt_score = 1.0
    elif avg_response_h <= 24:
        rt_score = 0.85
    elif avg_response_h <= 72:
        rt_score = 0.65
    elif avg_response_h <= 168:
        rt_score = 0.45
    else:
        rt_score = 0.2

    # 5. Market validation
    saved = sig.get("saved_by_recruiters_30d", 0)
    views = sig.get("profile_views_received_30d", 0)
    search_app = sig.get("search_appearance_30d", 0)
    market_score = min(1.0, (math.log1p(saved) / 4 + math.log1p(views) / 6 +
                              math.log1p(search_app) / 8) / 3 * 1.5)

    # 6. Interview completion rate
    icr = sig.get("interview_completion_rate", 0.7)
    interview_score = icr

    # 7. Profile authenticity
    completeness = sig.get("profile_completeness_score", 70) / 100
    verified = (sig.get("verified_email", False) + sig.get("verified_phone", False) +
                sig.get("linkedin_connected", False)) / 3
    github = max(0, sig.get("github_activity_score", 0)) / 100
    authenticity = (completeness * 0.4 + verified * 0.4 + github * 0.2)

    behavioral_score = (
        0.20 * recency_score +
        0.15 * open_flag +
        0.20 * response_score +
        0.10 * rt_score +
        0.15 * market_score +
        0.10 * interview_score +
        0.10 * authenticity
    )
    return min(1.0, behavioral_score)


# ---------------------------------------------------------------------------
# Location & Logistics Score
# ---------------------------------------------------------------------------

def score_location(candidate: dict) -> float:
    """
    Score based on location fit for Pune/Noida/India-based role.
    """
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    notice_period = signals.get("notice_period_days", 60)
    preferred_mode = signals.get("preferred_work_mode", "hybrid")

    # Country/city score
    if country == "india":
        # Check for preferred cities
        if any(city in location for city in INDIA_TIER1_CITIES):
            city_score = 1.0
        else:
            city_score = 0.8  # Other India cities
    else:
        # Outside India — JD says case-by-case, no visa sponsorship
        if willing_to_relocate:
            city_score = 0.35
        else:
            city_score = 0.15

    # Notice period score (JD says sub-30 ideal, can buy out 30 days)
    if notice_period <= 15:
        notice_score = 1.0
    elif notice_period <= 30:
        notice_score = 0.90
    elif notice_period <= 45:
        notice_score = 0.70
    elif notice_period <= 60:
        notice_score = 0.55
    elif notice_period <= 90:
        notice_score = 0.40
    else:
        notice_score = 0.25

    # Work mode alignment (JD says hybrid-flexible)
    mode_score = 1.0 if preferred_mode in ("hybrid", "flexible") else 0.75

    loc_score = 0.55 * city_score + 0.35 * notice_score + 0.10 * mode_score
    return min(1.0, loc_score)


# ---------------------------------------------------------------------------
# Red Flag Multipliers
# ---------------------------------------------------------------------------

def compute_red_flags(candidate: dict, role_tier: str) -> tuple[float, list[str]]:
    """
    Compute multiplicative penalty for hard disqualifiers.
    Returns (multiplier 0-1, list of flag descriptions).
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    multiplier = 1.0
    flags = []

    # 1. Non-technical role
    if role_tier == "non_technical":
        multiplier *= 0.05
        flags.append("non_technical_role")

    # 2. Entire career at consulting giants
    companies = [j.get("company", "").lower().strip() for j in career]
    is_all_consulting = len(companies) > 0 and all(
        any(cg in c for cg in CONSULTING_GIANTS) for c in companies
    )
    if is_all_consulting:
        multiplier *= 0.35
        flags.append("pure_consulting_career")

    # 3. Outside India + not willing to relocate
    country = profile.get("country", "").lower()
    willing = signals.get("willing_to_relocate", False)
    if country != "india" and not willing:
        multiplier *= 0.25
        flags.append("outside_india_no_relocation")

    # 4. Only very recent AI experience (<12 months LLM-only)
    recent_ai_months = 0
    pre_llm_ai_months = 0
    for job in career:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        months = job.get("duration_months", 0)
        is_recent_llm_only = ("langchain" in desc or "chatgpt" in desc or "gpt-4" in desc) and \
                              not any(t in desc for t in {"embedding", "retrieval", "ranking",
                                                          "recommendation", "vector"})
        is_pre_llm_ai = any(t in desc or t in title for t in
                            {"recommendation", "retrieval", "ranking", "embedding",
                             "machine learning", "deep learning", "nlp"})
        if is_recent_llm_only:
            recent_ai_months += months
        if is_pre_llm_ai:
            pre_llm_ai_months += months

    if recent_ai_months > 0 and pre_llm_ai_months == 0:
        multiplier *= 0.5
        flags.append("only_recent_llm_tutorial_experience")

    return multiplier, flags


# ---------------------------------------------------------------------------
# Reasoning Generator
# ---------------------------------------------------------------------------

def generate_reasoning(candidate: dict, scores: dict, role_tier: str,
                       matched_groups: set, flags: list) -> str:
    """
    Generate a specific, honest 1-2 sentence reasoning for this candidate's rank.
    """
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    company = profile.get("current_company", "Unknown")
    response_rate = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 60)
    open_to_work = signals.get("open_to_work_flag", False)

    # Key matched skills
    group_display = {
        "embeddings": "embeddings/retrieval",
        "vector_db": "vector DB",
        "ranking_retrieval": "ranking/IR",
        "python_ml": "Python/ML",
        "llm_finetuning": "LLM fine-tuning",
        "mlops_eval": "MLOps/eval",
        "production_signals": "production deployment",
    }
    matched_display = [group_display.get(g, g) for g in matched_groups
                       if g in {"embeddings", "vector_db", "ranking_retrieval"}]

    final_score = scores["final"]
    tech_score = scores["technical"]

    if flags and "non_technical_role" in flags:
        return (f"{title} ({yoe:.0f}y at {company}) — wrong domain for this JD; "
                f"technical skills don't compensate for non-AI/ML career trajectory.")

    if flags and "pure_consulting_career" in flags:
        return (f"{title} with {yoe:.1f}y experience, but entire career at IT services firms; "
                f"JD explicitly seeks product-company experience.")

    if flags and "outside_india_no_relocation" in flags:
        loc = profile.get("location", "")
        return (f"{title} based in {loc} with no relocation intent; "
                f"role is India-based and company doesn't sponsor visas.")

    # Compose honest reasoning
    parts = []
    if role_tier == "strong_ai":
        parts.append(f"{title} with {yoe:.1f}y in AI/ML at {company}")
    else:
        parts.append(f"{title} ({yoe:.1f}y) at {company}")

    if matched_display:
        parts.append(f"skills cover {', '.join(matched_display)}")

    if tech_score >= 0.7:
        parts.append("strong technical alignment with embedding/retrieval JD requirements")
    elif tech_score >= 0.45:
        parts.append("moderate technical fit with some key retrieval/ranking skills")
    else:
        parts.append("limited direct skill overlap with JD core requirements")

    # Behavioral note
    if not open_to_work:
        parts.append("not marked open-to-work")
    if response_rate < 0.2:
        parts.append(f"low recruiter response rate ({response_rate:.0%})")

    if notice > 90:
        parts.append(f"long notice period ({notice}d)")
    elif notice <= 30:
        parts.append(f"short notice ({notice}d, JD-preferred)")

    sent1 = f"{parts[0]}; {'; '.join(parts[1:3])}."
    sent2 = ""
    if len(parts) > 3:
        sent2 = f" {'; '.join(parts[3:]).capitalize()}."

    return (sent1 + sent2).strip()


# ---------------------------------------------------------------------------
# Main Ranking Pipeline
# ---------------------------------------------------------------------------

WEIGHTS = {
    "role": 0.20,
    "technical": 0.30,
    "career": 0.20,
    "semantic": 0.15,
    "behavioral": 0.10,
    "location": 0.05,
}

def score_candidate(candidate: dict) -> dict:
    """
    Compute all sub-scores and the final weighted score for one candidate.
    """
    # Honeypot check
    if detect_honeypot(candidate):
        return {
            "candidate_id": candidate["candidate_id"],
            "final": 0.0,
            "honeypot": True,
            "flags": ["honeypot"],
            "role": 0.0, "technical": 0.0, "career": 0.0,
            "semantic": 0.0, "behavioral": 0.0, "location": 0.0,
            "role_tier": "honeypot", "matched_groups": set(),
        }

    role_score, role_tier = score_role_legitimacy(candidate)
    tech_score, matched_groups = score_skills(candidate)
    career_score = score_career_quality(candidate)
    semantic_score = score_semantic_fit(candidate)
    behavioral_score = score_behavioral(candidate)
    location_score = score_location(candidate)

    weighted = (
        WEIGHTS["role"] * role_score +
        WEIGHTS["technical"] * tech_score +
        WEIGHTS["career"] * career_score +
        WEIGHTS["semantic"] * semantic_score +
        WEIGHTS["behavioral"] * behavioral_score +
        WEIGHTS["location"] * location_score
    )

    multiplier, flags = compute_red_flags(candidate, role_tier)
    final = weighted * multiplier

    return {
        "candidate_id": candidate["candidate_id"],
        "final": round(final, 6),
        "honeypot": False,
        "flags": flags,
        "role": role_score,
        "technical": tech_score,
        "career": career_score,
        "semantic": semantic_score,
        "behavioral": behavioral_score,
        "location": location_score,
        "role_tier": role_tier,
        "matched_groups": matched_groups,
    }


def run(candidates_path: str, output_path: str) -> None:
    import time
    t0 = time.time()

    print(f"Loading candidates from: {candidates_path}", file=sys.stderr)

    # Stream the JSONL file
    scored = []
    n = 0

    opener = open
    if candidates_path.endswith(".gz"):
        import gzip
        opener = lambda p, **kw: gzip.open(p, "rt", **kw)

    with opener(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = score_candidate(candidate)
            scored.append(result)
            n += 1

            if n % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {n:,} candidates in {elapsed:.1f}s", file=sys.stderr)

    t1 = time.time()
    print(f"Scored {n:,} candidates in {t1-t0:.1f}s", file=sys.stderr)

    # Sort by final score descending
    scored.sort(key=lambda x: -x["final"])

    # Take top 100 — but first load full candidate data for reasoning
    top100_ids = {r["candidate_id"] for r in scored[:100]}

    # Second pass: reload the top-100 candidates for reasoning generation
    candidates_map = {}
    with opener(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                if c["candidate_id"] in top100_ids:
                    candidates_map[c["candidate_id"]] = c
            except json.JSONDecodeError:
                continue

    # Write output CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize scores to [0, 1] range within top 100
    top100 = scored[:100]
    max_score = top100[0]["final"] if top100 else 1.0
    min_score = top100[-1]["final"] if top100 else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, result in enumerate(top100, start=1):
            cid = result["candidate_id"]
            raw_score = result["final"]
            # Normalize to [0.400, 0.995] range for output
            norm_score = 0.400 + (raw_score - min_score) / score_range * 0.595
            norm_score = round(norm_score, 4)

            candidate = candidates_map.get(cid, {})
            if candidate:
                reasoning = generate_reasoning(
                    candidate, result, result["role_tier"],
                    result["matched_groups"], result["flags"]
                )
            else:
                reasoning = f"Rank {rank}: scored {raw_score:.4f}."

            writer.writerow([cid, rank, norm_score, reasoning])

    t2 = time.time()
    print(f"Written top-100 to: {output_path} (total: {t2-t0:.1f}s)", file=sys.stderr)

    # Print summary stats
    non_hp = [r for r in scored if not r["honeypot"]]
    hp_count = n - len(non_hp)
    top10_tiers = [scored[i]["role_tier"] for i in range(min(10, len(scored)))]
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Total candidates: {n:,}", file=sys.stderr)
    print(f"  Honeypots detected: {hp_count}", file=sys.stderr)
    print(f"  Top-10 role tiers: {top10_tiers}", file=sys.stderr)
    print(f"  Score range (top-100): [{min_score:.4f}, {max_score:.4f}]", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Candidate Ranker")
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    run(args.candidates, args.out)
