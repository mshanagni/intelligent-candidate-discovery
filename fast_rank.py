"""
fast_rank.py — Optimized single-pass candidate scorer for Redrob Hackathon.

Produces top-100 CSV in <45s on CPU for 100K candidates.
v2: +company prestige, +education quality, +quantified impact,
    +skill recency, +career progression signals.

Usage:
  python fast_rank.py --candidates candidates.jsonl --out submission.csv
"""

import argparse, csv, heapq, math, re, sys
from datetime import date, datetime

try:
    import orjson as _json_lib
    def _loads(b): return _json_lib.loads(b)
except ImportError:
    import json as _json_lib
    def _loads(b): return _json_lib.loads(b)

# ── Skill map ─────────────────────────────────────────────────────────────────
SKILL_MAP = {
    "embeddings":2.5,"sentence-transformers":2.5,"sentence transformers":2.5,
    "bge":2.5,"e5":2.5,"openai embeddings":2.5,"semantic search":2.5,
    "text embeddings":2.5,"bi-encoder":2.5,"cross-encoder":2.5,
    "dense retrieval":2.5,"embedding":2.5,
    "pinecone":2.5,"weaviate":2.5,"qdrant":2.5,"milvus":2.5,"faiss":2.5,
    "opensearch":2.5,"elasticsearch":2.5,"pgvector":2.5,"annoy":2.5,
    "vector database":2.5,"vector store":2.5,"hybrid search":2.5,
    "approximate nearest neighbor":2.5,"ann":2.5,
    "ranking":2.0,"retrieval":2.0,"information retrieval":2.0,"bm25":2.0,
    "learning to rank":2.0,"ltr":2.0,"reranking":2.0,"re-ranking":2.0,
    "ndcg":2.0,"mrr":2.0,"recommendation systems":2.0,"recommender systems":2.0,
    "search engine":2.0,"haystack":2.0,"llamaindex":2.0,"llama index":2.0,
    "xgboost":1.8,"lightgbm":1.8,"gradient boosting":1.8,
    "python":1.5,"pytorch":1.5,"tensorflow":1.5,"scikit-learn":1.5,"sklearn":1.5,
    "hugging face":1.5,"hugging face transformers":1.5,"huggingface":1.5,
    "transformers":1.5,"numpy":1.2,"pandas":1.2,
    "fine-tuning llms":1.2,"fine tuning llms":1.2,"lora":1.2,"qlora":1.2,
    "peft":1.2,"instruction tuning":1.2,"rlhf":1.2,"langchain":1.0,
    "mlflow":1.0,"weights & biases":1.0,"wandb":1.0,"kubeflow":0.9,
    "bentoml":0.9,"triton":0.9,"torchserve":0.9,
}
# Synonym normalisation (maps variant → canonical key in SKILL_MAP)
SKILL_SYNONYMS = {
    "hf":"hugging face","hf transformers":"hugging face transformers",
    "tf":"tensorflow","sk-learn":"scikit-learn","xgb":"xgboost","lgbm":"lightgbm",
    "rag":"retrieval","vector db":"vector database","vector search":"vector database",
    "llama-index":"llamaindex","haystack2":"haystack","openai api":"langchain",
}

PROF_W = {"expert":1.0,"advanced":0.85,"intermediate":0.60,"beginner":0.30}
MAX_SKILL_WEIGHT = 2.5*3 + 2.0 + 1.5 + 1.2 + 1.0

# ── JD signal sets ────────────────────────────────────────────────────────────
JD_WORDS = frozenset({
    "embeddings","embedding","retrieval","ranking","recommendation","search",
    "vector","pinecone","faiss","milvus","weaviate","qdrant","elasticsearch",
    "opensearch","bm25","ndcg","mrr","map","python","pytorch","transformers",
    "bert","llm","fine-tuning","lora","peft","production","deployed","shipped",
    "scale","encoder","decoder","semantic","dense","sparse","haystack",
    "llamaindex","mlflow","wandb","xgboost","lightgbm","huggingface","sklearn",
    "nlp","reranking","inference","latency","index","ann","hybrid",
})
JD_BIGRAMS = frozenset({
    "embedding retrieval","vector database","hybrid search","dense retrieval",
    "ranking system","recommendation system","search engine","information retrieval",
    "learning rank","fine tuning","large scale","real users","a/b testing",
    "semantic search","candidate ranking",
})
NEG_WORDS = frozenset({
    "computer vision","image classification","object detection","yolo","opencv",
    "speech recognition","text-to-speech","tts","robotics","android","ios",
    "flutter","react native","accounting","tally","sap","salesforce","crm",
    "marketing","seo","content writing","photoshop","illustrator",
    "civil","mechanical","electrical","procurement","logistics","supply chain",
})

# ── Role sets ─────────────────────────────────────────────────────────────────
STRONG_AI = frozenset({
    "ml engineer","machine learning engineer","ai engineer","research scientist",
    "nlp engineer","data scientist","applied scientist","research engineer",
    "recommendation systems engineer","recommendation engineer","search engineer",
    "retrieval engineer","ranking engineer","ai/ml engineer","deep learning engineer",
    "llm engineer","senior ml","staff ml","principal ml","senior ai","staff ai",
    "senior data scientist","lead data scientist","principal data scientist",
    "senior research","lead ml","lead ai","applied ml engineer",
})
MOD_TECH = frozenset({
    "software engineer","backend engineer","data engineer","cloud engineer",
    "platform engineer","infrastructure engineer","devops engineer","sre",
    "solutions architect","tech lead","senior engineer","principal engineer",
    "staff engineer","software developer",
})
NON_TECH = frozenset({
    "hr manager","human resources","marketing manager","content writer",
    "accountant","graphic designer","customer support","operations manager",
    "sales","business analyst","project manager","civil engineer",
    "mechanical engineer","electrical engineer","product manager",
    "scrum master","finance","recruiter","procurement","supply chain",
    "logistics","legal","compliance","administrator","admin",
    "java developer",".net developer","sap consultant",
    "mobile developer","android developer","ios developer",
    "ui designer","ux designer","frontend engineer","web developer",
    "qa engineer","quality assurance","test engineer","frontend developer",
})
AI_WORDS = frozenset({
    "machine learning","ml engineer","ai engineer","nlp","deep learning",
    "neural network","retrieval","ranking","recommendation","embedding",
    "llm","language model","vector","search engine","transformer","bert",
    "gpt","fine-tuning","rag","reranking","semantic search","data scientist",
})
CONSULTING = frozenset({
    "tcs","infosys","wipro","accenture","cognizant","capgemini",
    "hcl","tech mahindra","mindtree","mphasis","ltimindtree",
    "hexaware","niit","kpit","persistent systems","cyient","zensar","birlasoft",
})
INDIA_CITIES = frozenset({
    "pune","noida","hyderabad","mumbai","delhi","bangalore","bengaluru",
    "gurgaon","gurugram","chennai","kolkata","ahmedabad","greater noida",
})

# ── NEW: Company prestige tiers ───────────────────────────────────────────────
CO_TIER1 = frozenset({
    "google","meta","amazon","microsoft","apple","netflix","deepmind","openai",
    "anthropic","nvidia","linkedin","uber","airbnb","stripe","databricks",
})
CO_TIER2 = frozenset({
    "flipkart","zomato","swiggy","cred","meesho","razorpay","phonepe","paytm",
    "ola","sarvam","adobe","atlassian","salesforce","twitter","bytedance",
    "rephrase","freshworks","cleartax","zerodha","groww","nykaa","sharechat",
    "moj","dailyhunt","slice","jar","juspay","healthifyme",
})
CO_TIER3 = frozenset({
    "byju","unacademy","vedantu","limeroad","shopclues","snapdeal",
    "indiamart","justdial","makemytrip","oyo","ixigo",
})

# ── NEW: Elite universities ───────────────────────────────────────────────────
ELITE_EDU = frozenset({
    "iit","iim","bits","nit","iisc","iiser","iiit",
    "stanford","mit","cmu","carnegie mellon","berkeley","oxford","cambridge",
    "eth zurich","waterloo","toronto","montreal","nyu","columbia","yale","princeton",
})

# ── NEW: Quantified impact regex ──────────────────────────────────────────────
_METRIC_RE = re.compile(
    r'\d+[%xX×]\s*(?:improvement|reduction|increase|faster|lift|gain)|'
    r'(?:million|billion|crore|lakh)\s*(?:users|requests|queries|impressions)|'
    r'p\d{2}\s*latency|latency.*?\d+\s*ms|throughput.*?\d+|'
    r'\d+\s*(?:ms|milliseconds?)\s*(?:latency|p99|p95)|'
    r'serving\s*\d+|ranked\s*\d+|retrieved\s*\d+',
    re.IGNORECASE
)

# ── NEW: Seniority levels for progression scoring ─────────────────────────────
_SENIORITY = {
    "intern":0,"junior":1,"associate":2,"mid":3,"senior":4,
    "lead":5,"staff":6,"principal":7,"director":8,"vp":9,
}

TODAY = date.today()


# ── Honeypot ──────────────────────────────────────────────────────────────────
def _honeypot(c):
    yoe = c["profile"].get("years_of_experience", 0)
    edu = c.get("education", [])
    if edu:
        try:
            gy = min(e.get("end_year",9999) for e in edu if e.get("end_year"))
            if gy > TODAY.year or gy < 1960: return True
            if yoe > (TODAY.year - gy + 2): return True
        except Exception: pass
    career = c.get("career_history", [])
    total_m = sum(j.get("duration_months",0) for j in career)
    if total_m > (yoe + 5) * 12: return True
    for j in career:
        sd = j.get("start_date","")
        if sd:
            try:
                sy = int(sd[:4])
                if sy > TODAY.year or sy < 1970: return True
            except Exception: pass
    return False


# ── NEW helpers ───────────────────────────────────────────────────────────────

def _company_prestige(company_names):
    """Return best prestige score (0.4–1.0) across all career companies."""
    best = 0.4
    for cn in company_names:
        if any(t in cn for t in CO_TIER1): best = max(best, 1.00)
        elif any(t in cn for t in CO_TIER2): best = max(best, 0.85)
        elif any(t in cn for t in CO_TIER3): best = max(best, 0.70)
        elif not any(cg in cn for cg in CONSULTING): best = max(best, 0.60)
    return best

def _education_score(edu):
    """Elite university = 1.0, unknown = 0.55, missing = 0.5."""
    if not edu: return 0.5
    for e in edu:
        inst = (e.get("institution") or e.get("school") or "").lower()
        if any(u in inst for u in ELITE_EDU): return 1.0
    return 0.55

def _quantified_impact(career):
    """Count job descriptions with measurable metrics."""
    hits = sum(1 for j in career
               if _METRIC_RE.search(j.get("description","") or ""))
    return min(1.0, hits / 3)

def _career_progression(career_title_l):
    """Upward seniority trajectory = 1.0, flat = 0.6, downward = 0.3."""
    if len(career_title_l) < 2: return 0.6
    levels = []
    for t in career_title_l:
        for kw, lv in _SENIORITY.items():
            if kw in t: levels.append(lv); break
        else: levels.append(3)  # unknown → mid-level
    # Compare first half avg vs second half avg
    mid = len(levels) // 2
    early = sum(levels[:mid]) / max(mid, 1)
    late  = sum(levels[mid:]) / max(len(levels)-mid, 1)
    if late > early + 0.5: return 1.0    # clear upward
    if late < early - 0.5: return 0.3    # downward (job-hopping down)
    return 0.6                            # flat

def _skill_recency(career, skill_map):
    """Boost if high-value skills appear in recent (last 2) job descriptions."""
    if not career: return 1.0
    recent_text = " ".join(
        (j.get("description","") or "").lower()
        for j in career[-2:]
    )
    high_val = sum(1 for sk, wt in skill_map.items()
                   if wt >= 2.0 and sk in recent_text)
    return min(1.0, 0.7 + high_val * 0.1)


# ── Core scorer ───────────────────────────────────────────────────────────────

def _score(c):
    if _honeypot(c):
        return 0.0,"honeypot",set(),[],0.0,0.0,0.0,0.0,0.0,0.0

    profile  = c["profile"]
    career   = c.get("career_history", [])
    skills   = c.get("skills", [])
    sig      = c.get("redrob_signals", {})
    edu      = c.get("education", [])

    title    = profile.get("current_title","").lower()
    summary  = profile.get("summary","").lower()
    headline = profile.get("headline","").lower()
    yoe      = profile.get("years_of_experience", 0)
    country  = profile.get("country","").lower()
    location = profile.get("location","").lower()

    # ── Career analysis ──────────────────────────────────────────────────────
    career_texts  = []
    consulting_m  = total_m = prod_hits = ai_ml_m = 0
    is_pure_research = True
    company_names = []
    career_title_l = []

    PROD_KW = frozenset({"production","deployed","shipped","at scale","real users",
                         "live","online","latency","throughput","ranking","retrieval",
                         "search","recommendation","embedding","vector","index"})
    RESEARCH_T = frozenset({"research scientist","research engineer","phd",
                             "postdoc","professor","academic"})
    RESEARCH_I = frozenset({"academia","university","research"})

    for j in career:
        jt  = j.get("title","").lower()
        jco = j.get("company","").lower().strip()
        jd  = j.get("description","").lower()
        ji  = j.get("industry","").lower()
        m   = j.get("duration_months", 0)

        career_texts.append(jd)
        career_title_l.append(jt)
        company_names.append(jco)
        total_m += m

        if any(cg in jco for cg in CONSULTING) or "it services" in ji or "consulting" in ji:
            consulting_m += m

        prod_hits = min(prod_hits + sum(1 for kw in PROD_KW if kw in jd), 16)

        if any(a in jt for a in STRONG_AI) or any(a in jt for a in {"data scientist","ml","nlp","recommendation","search"}):
            ai_ml_m += m

        if not (any(rt in jt for rt in RESEARCH_T) or any(ri in ji for ri in RESEARCH_I)):
            is_pure_research = False

    full_career = " ".join(career_texts)
    skills_text = " ".join(s["name"].lower() for s in skills)
    all_titles  = [title] + career_title_l
    full_text   = f"{summary} {headline} {full_career} {skills_text}"

    # ── 1. Role Legitimacy ───────────────────────────────────────────────────
    role_tier = "unknown"; role_score = 0.0
    for t in all_titles:
        if any(a in t for a in STRONG_AI):
            role_score = 1.0; role_tier = "strong_ai"; break

    if role_tier == "unknown":
        ai_count = sum(1 for kw in AI_WORDS if kw in summary or kw in headline)
        for t in all_titles:
            if any(m in t for m in MOD_TECH):
                role_score = 0.75 if ai_count >= 2 else 0.55
                role_tier  = "moderate_tech_ai" if ai_count >= 2 else "moderate_tech"; break

    if role_tier == "unknown":
        ai_count = sum(1 for kw in AI_WORDS if kw in summary or kw in headline)
        for t in all_titles:
            if any(b in t for b in NON_TECH):
                role_score = 0.25 if ai_count >= 3 else 0.08
                role_tier  = "adjacent" if ai_count >= 3 else "non_technical"; break

    if role_tier == "unknown":
        ai_count = sum(1 for kw in AI_WORDS if kw in summary or kw in headline)
        role_score = min(0.65, 0.2 + ai_count * 0.1)
        role_tier  = "moderate_tech_ai" if ai_count >= 4 else "adjacent"

    # ── 2. Skills Score ──────────────────────────────────────────────────────
    group_scores = {}; matched_groups = set()
    sk_assess = sig.get("skill_assessment_scores", {})

    for sk in skills:
        nm  = sk["name"].lower().strip()
        nm  = SKILL_SYNONYMS.get(nm, nm)          # normalise synonyms
        wt  = SKILL_MAP.get(nm)
        if wt is None: continue
        prof = PROF_W.get(sk.get("proficiency","intermediate"), 0.5)
        end  = min(0.3, math.log1p(sk.get("endorsements",0)) / 15)
        dur  = min(0.2, sk.get("duration_months",0) / 60)
        ass  = (sk_assess.get(sk["name"],0) / 100) * 0.2
        raw  = prof * (1 + end + dur + ass)
        group_scores[wt] = max(group_scores.get(wt,0), min(1.0, raw))
        matched_groups.add(nm)

    skill_score = min(1.0, sum(wt*best for wt,best in group_scores.items()) / MAX_SKILL_WEIGHT)

    # Apply skill recency boost
    recency_mult = _skill_recency(career, SKILL_MAP)
    skill_score  = min(1.0, skill_score * recency_mult)

    # ── 3. Career Quality (enhanced) ─────────────────────────────────────────
    if   6  <= yoe <= 8:  yoe_s = 1.00
    elif 5  <= yoe <  6:  yoe_s = 0.85
    elif 8  < yoe <= 9:   yoe_s = 0.85
    elif 4  <= yoe <  5:  yoe_s = 0.65
    elif 9  < yoe <= 12:  yoe_s = 0.65
    elif 3  <= yoe <  4:  yoe_s = 0.45
    else:                 yoe_s = 0.25

    cons_ratio   = (consulting_m / total_m) if total_m else 0.5
    company_s    = 1.0 - cons_ratio * 0.7
    prestige_s   = _company_prestige(company_names)           # NEW
    prod_s       = min(1.0, prod_hits / 16)
    impact_s     = _quantified_impact(career)                 # NEW
    progression_s= _career_progression(career_title_l)       # NEW
    edu_s        = _education_score(edu)                      # NEW
    ai_prog      = min(1.0, (ai_ml_m / max(total_m,1)) * 2)
    research_p   = 0.5 if is_pure_research else 1.0

    career_score = (
        0.20 * yoe_s        +
        0.15 * company_s    +
        0.15 * prestige_s   +   # replaces pure consulting ratio weight
        0.15 * prod_s       +
        0.10 * impact_s     +   # NEW: quantified metrics in descriptions
        0.10 * progression_s+   # NEW: upward trajectory
        0.10 * edu_s        +   # NEW: elite university bonus
        0.05 * ai_prog
    ) * research_p

    # ── 4. Semantic Fit ──────────────────────────────────────────────────────
    words_set  = set(full_text.split())
    pos_count  = len(JD_WORDS & words_set)
    pos_score  = min(1.0, pos_count / 12)
    bg_count   = sum(1 for bg in JD_BIGRAMS if bg in full_text)
    bg_bonus   = min(0.3, bg_count / 8)
    neg_count  = sum(1 for nw in NEG_WORDS if nw in full_text)
    neg_pen    = min(0.4, neg_count * 0.08)
    semantic_score = max(0.0, pos_score + bg_bonus - neg_pen)

    # ── 5. Behavioral ────────────────────────────────────────────────────────
    la = sig.get("last_active_date","")
    try:
        di  = (TODAY - datetime.strptime(la[:10], "%Y-%m-%d").date()).days
        rec = 1.0 if di<=30 else 0.8 if di<=90 else 0.5 if di<=180 else 0.3 if di<=365 else 0.1
    except Exception:
        rec = 0.4

    open_w  = 1.0 if sig.get("open_to_work_flag", False) else 0.4
    rr      = sig.get("recruiter_response_rate", 0.5)
    arh     = sig.get("avg_response_time_hours", 48)
    rt_s    = 1.0 if arh<=4 else 0.85 if arh<=24 else 0.65 if arh<=72 else 0.45 if arh<=168 else 0.2
    saved   = sig.get("saved_by_recruiters_30d", 0)
    views   = sig.get("profile_views_received_30d", 0)
    search  = sig.get("search_appearance_30d", 0)
    mkt_s   = min(1.0, (math.log1p(saved)/4 + math.log1p(views)/6 + math.log1p(search)/8)/3*1.5)
    icr     = sig.get("interview_completion_rate", 0.7)
    comp    = sig.get("profile_completeness_score", 70) / 100
    ver     = (sig.get("verified_email",False) + sig.get("verified_phone",False) +
               sig.get("linkedin_connected",False)) / 3
    gh      = max(0, sig.get("github_activity_score", 0)) / 100
    auth    = comp*0.4 + ver*0.4 + gh*0.2

    behavioral_score = (0.20*rec + 0.15*open_w + 0.20*rr + 0.10*rt_s +
                        0.15*mkt_s + 0.10*icr + 0.10*auth)

    # ── 6. Location ──────────────────────────────────────────────────────────
    reloc  = sig.get("willing_to_relocate", False)
    notice = sig.get("notice_period_days", 60)
    mode   = sig.get("preferred_work_mode","hybrid")

    city_s = (1.0 if country=="india" and any(ci in location for ci in INDIA_CITIES)
              else 0.8 if country=="india"
              else 0.35 if reloc else 0.15)
    not_s  = (1.0 if notice<=15 else 0.90 if notice<=30 else 0.70 if notice<=45
              else 0.55 if notice<=60 else 0.40 if notice<=90 else 0.25)
    mode_s = 1.0 if mode in ("hybrid","flexible") else 0.75
    location_score = 0.55*city_s + 0.35*not_s + 0.10*mode_s

    # ── Weighted composite (updated weights) ─────────────────────────────────
    composite = (0.20*role_score   +
                 0.28*skill_score  +   # slight drop: skill recency already boosts
                 0.25*career_score +   # increase: now includes prestige/edu/impact
                 0.12*semantic_score+  # slight drop
                 0.10*behavioral_score +
                 0.05*location_score)

    # ── Red-flag multipliers ─────────────────────────────────────────────────
    flags = []; mult = 1.0

    if role_tier == "non_technical":
        mult *= 0.05; flags.append("non_technical")

    if company_names and all(any(cg in cn for cg in CONSULTING) for cn in company_names):
        mult *= 0.35; flags.append("pure_consulting")

    if country != "india" and not reloc:
        mult *= 0.25; flags.append("outside_india_no_reloc")

    recent_llm = "langchain" in full_career or "chatgpt" in full_career
    pre_llm_ai = any(w in full_career for w in
                     ("recommendation","retrieval","ranking","embedding","machine learning"))
    if recent_llm and not pre_llm_ai:
        mult *= 0.5; flags.append("only_recent_llm")

    final = composite * mult
    return final, role_tier, matched_groups, flags, role_score, skill_score, career_score, semantic_score, behavioral_score, location_score


# ── Reasoning ─────────────────────────────────────────────────────────────────
_GROUP_DISPLAY = {
    "pinecone":"Pinecone","weaviate":"Weaviate","qdrant":"Qdrant","milvus":"Milvus",
    "faiss":"FAISS","opensearch":"OpenSearch","elasticsearch":"Elasticsearch",
    "embeddings":"embeddings","sentence-transformers":"sentence-transformers",
    "ranking":"ranking","retrieval":"retrieval","recommendation systems":"RecSys",
    "python":"Python","pytorch":"PyTorch","transformers":"Transformers",
    "fine-tuning llms":"LLM fine-tuning","lora":"LoRA","peft":"PEFT",
    "mlflow":"MLflow","weights & biases":"W&B",
}

def _reasoning(c, result):
    final, role_tier, matched, flags, r_s, t_s, ca_s, se_s, be_s, loc_s = result
    p   = c["profile"]
    sig = c.get("redrob_signals", {})

    title  = p.get("current_title","?")
    yoe    = p.get("years_of_experience",0)
    co     = p.get("current_company","?")
    loc    = p.get("location","?")
    