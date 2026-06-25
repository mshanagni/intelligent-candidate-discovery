"""
fast_rank.py — Optimized single-pass candidate scorer for Redrob Hackathon.

Produces top-100 CSV in <45s on CPU for 100K candidates.
All scoring is inline (no module hops) using fast set operations.

Usage:
  python fast_rank.py --candidates candidates.jsonl --out submission.csv
"""

import argparse, csv, heapq, math, sys
from datetime import date, datetime

try:
    import orjson as _json_lib
    def _loads(b): return _json_lib.loads(b)
except ImportError:
    import json as _json_lib
    def _loads(b): return _json_lib.loads(b)

# ── JD-derived lookup sets ────────────────────────────────────────────────────

# Skills that map directly to core groups (used for fast dict lookup)
SKILL_MAP = {
    # embeddings (weight 2.5)
    "embeddings":2.5,"sentence-transformers":2.5,"sentence transformers":2.5,
    "bge":2.5,"e5":2.5,"openai embeddings":2.5,"semantic search":2.5,
    "text embeddings":2.5,"bi-encoder":2.5,"cross-encoder":2.5,
    "dense retrieval":2.5,"embedding":2.5,
    # vector_db (weight 2.5)
    "pinecone":2.5,"weaviate":2.5,"qdrant":2.5,"milvus":2.5,"faiss":2.5,
    "opensearch":2.5,"elasticsearch":2.5,"pgvector":2.5,"annoy":2.5,
    "vector database":2.5,"vector store":2.5,"hybrid search":2.5,
    "approximate nearest neighbor":2.5,"ann":2.5,
    # ranking/IR (weight 2.0)
    "ranking":2.0,"retrieval":2.0,"information retrieval":2.0,"bm25":2.0,
    "learning to rank":2.0,"ltr":2.0,"reranking":2.0,"re-ranking":2.0,
    "ndcg":2.0,"mrr":2.0,"recommendation systems":2.0,"recommender systems":2.0,
    "search engine":2.0,"haystack":2.0,"llamaindex":2.0,"llama index":2.0,
    "xgboost":1.8,"lightgbm":1.8,"gradient boosting":1.8,
    # Python/ML (weight 1.5)
    "python":1.5,"pytorch":1.5,"tensorflow":1.5,"scikit-learn":1.5,"sklearn":1.5,
    "hugging face":1.5,"hugging face transformers":1.5,"huggingface":1.5,
    "transformers":1.5,"numpy":1.2,"pandas":1.2,
    # LLM fine-tuning (weight 1.2)
    "fine-tuning llms":1.2,"fine tuning llms":1.2,"lora":1.2,"qlora":1.2,
    "peft":1.2,"instruction tuning":1.2,"rlhf":1.2,"langchain":1.0,
    # MLOps/eval (weight 1.0)
    "mlflow":1.0,"weights & biases":1.0,"wandb":1.0,"kubeflow":0.9,
    "bentoml":0.9,"triton":0.9,"torchserve":0.9,
}

PROF_W = {"expert":1.0,"advanced":0.85,"intermediate":0.60,"beginner":0.30}
MAX_SKILL_WEIGHT = 2.5 * 3 + 2.0 + 1.5 + 1.2 + 1.0  # denominator for normalization

# JD positive signal words (fast set membership)
JD_WORDS = frozenset({
    "embeddings","embedding","retrieval","ranking","recommendation","search",
    "vector","pinecone","faiss","milvus","weaviate","qdrant","elasticsearch",
    "opensearch","bm25","ndcg","mrr","map","python","pytorch","transformers",
    "bert","llm","fine-tuning","lora","peft","production","deployed","shipped",
    "scale","encoder","decoder","semantic","dense","sparse","haystack",
    "llamaindex","mlflow","wandb","xgboost","lightgbm","huggingface","sklearn",
    "nlp","information","retrieval","reranking","inference","latency","index",
    "ann","approximate","nearest","neighbor","hybrid",
})
JD_BIGRAMS = frozenset({
    "embedding retrieval","vector database","hybrid search","dense retrieval",
    "ranking system","recommendation system","search engine","information retrieval",
    "learning rank","fine tuning","large scale","real users","a/b testing",
    "semantic search","candidate ranking",
})

# Negative domain words (CV/speech/robotics/non-tech)
NEG_WORDS = frozenset({
    "computer vision","image classification","object detection","yolo","opencv",
    "speech recognition","text-to-speech","tts","robotics","arduino","raspi",
    "android","ios","flutter","react native","swift","kotlin",
    "accounting","tally","sap","salesforce","crm","oracle","erp",
    "marketing","seo","sem","content writing","photoshop","illustrator","figma",
    "civil","mechanical","electrical","procurement","logistics","supply chain",
})

# Strong AI titles → role score 1.0
STRONG_AI = frozenset({
    "ml engineer","machine learning engineer","ai engineer","research scientist",
    "nlp engineer","data scientist","applied scientist","research engineer",
    "recommendation systems engineer","recommendation engineer","search engineer",
    "retrieval engineer","ranking engineer","ai/ml engineer","deep learning engineer",
    "llm engineer","senior ml","staff ml","principal ml","senior ai","staff ai",
    "senior data scientist","lead data scientist","principal data scientist",
    "senior research","lead ml","lead ai",
})
# Moderate tech titles → role score 0.55-0.75
MOD_TECH = frozenset({
    "software engineer","backend engineer","data engineer","cloud engineer",
    "platform engineer","infrastructure engineer","devops engineer","sre",
    "solutions architect","tech lead","senior engineer","principal engineer",
    "staff engineer","software developer",
})
# Non-technical → role score 0.08
NON_TECH = frozenset({
    "hr manager","human resources","marketing manager","content writer",
    "accountant","graphic designer","customer support","operations manager",
    "sales","business analyst","project manager","civil engineer",
    "mechanical engineer","electrical engineer","product manager",
    "scrum master","finance","recruiter","procurement","supply chain",
    "logistics","legal","compliance","administrator","admin",
    "java developer",".net developer","sap consultant","oracle",
    "mobile developer","android developer","ios developer",
    "ui designer","ux designer","frontend engineer","web developer",
    "qa engineer","quality assurance","test engineer","frontend developer",
})
# AI keyword signals (for nuanced role scoring)
AI_WORDS = frozenset({
    "machine learning","ml engineer","ai engineer","nlp","deep learning",
    "neural network","retrieval","ranking","recommendation","embedding",
    "llm","language model","vector","search engine","transformer","bert",
    "gpt","fine-tuning","rag","reranking","semantic search","data scientist",
})
# Consulting giants (penalized if entire career)
CONSULTING = frozenset({
    "tcs","infosys","wipro","accenture","cognizant","capgemini",
    "hcl","tech mahindra","mindtree","mphasis","ltimindtree",
    "hexaware","niit","kpit","persistent systems","cyient","zensar","birlasoft",
})
# Preferred India cities
INDIA_CITIES = frozenset({
    "pune","noida","hyderabad","mumbai","delhi","bangalore","bengaluru",
    "gurgaon","gurugram","chennai","kolkata","ahmedabad","greater noida",
})

TODAY = date.today()

# ── Fast honeypot detection ───────────────────────────────────────────────────

def _honeypot(c):
    yoe = c["profile"].get("years_of_experience", 0)
    edu = c.get("education", [])
    if edu:
        try:
            gy = min(e.get("end_year", 9999) for e in edu if e.get("end_year"))
            if gy > TODAY.year or gy < 1960:
                return True
            if yoe > (TODAY.year - gy + 2):
                return True
        except Exception:
            pass
    career = c.get("career_history", [])
    total_m = sum(j.get("duration_months", 0) for j in career)
    if total_m > (yoe + 5) * 12:
        return True
    for j in career:
        sd = j.get("start_date", "")
        if sd:
            try:
                sy = int(sd[:4])
                if sy > TODAY.year or sy < 1970:
                    return True
            except Exception:
                pass
    return False

# ── Core scorer (all in one flat function) ───────────────────────────────────

def _score(c):
    if _honeypot(c):
        return 0.0, "honeypot", set(), [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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

    # ── Career text (built once) ─────────────────────────────────────────────
    career_texts = []
    consulting_m = product_m = total_m = 0
    prod_hits = 0
    ai_ml_m = 0
    is_pure_research = True
    company_names = []
    career_title_l = []

    PROD_KW = frozenset({"production","deployed","shipped","at scale","real users",
                         "live","online","latency","throughput","ranking","retrieval",
                         "search","recommendation","embedding","vector","index"})

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

        is_cons = any(cg in jco for cg in CONSULTING)
        is_it   = "it services" in ji or "consulting" in ji
        if is_cons or is_it:
            consulting_m += m
        else:
            product_m += m

        # production signals
        for kw in PROD_KW:
            if kw in jd:
                prod_hits += 1
        prod_hits = min(prod_hits, 16)  # cap

        # AI/ML title check
        is_ai = any(a in jt for a in STRONG_AI) or any(a in jt for a in {"data scientist","ml","nlp","recommendation","search"})
        if is_ai:
            ai_ml_m += m

        # Research-only check
        RESEARCH_T = frozenset({"research scientist","research engineer","phd","postdoc","professor","academic"})
        RESEARCH_I = frozenset({"academia","university","research"})
        if not (any(rt in jt for rt in RESEARCH_T) or any(ri in ji for ri in RESEARCH_I)):
            is_pure_research = False

    full_career = " ".join(career_texts)
    skills_text = " ".join(s["name"].lower() for s in skills)
    all_titles  = [title] + career_title_l
    full_text   = f"{summary} {headline} {full_career} {skills_text}"

    # ── 1. Role Legitimacy Score ─────────────────────────────────────────────
    role_tier = "unknown"
    role_score = 0.0

    for t in all_titles:
        if any(a in t for a in STRONG_AI):
            role_score = 1.0; role_tier = "strong_ai"; break

    if role_tier == "unknown":
        ai_count = sum(1 for kw in AI_WORDS if kw in summary or kw in headline)
        for t in all_titles:
            if any(m in t for m in MOD_TECH):
                role_score = 0.75 if ai_count >= 2 else 0.55
                role_tier = "moderate_tech_ai" if ai_count >= 2 else "moderate_tech"
                break

    if role_tier == "unknown":
        ai_count = sum(1 for kw in AI_WORDS if kw in summary or kw in headline)
        for t in all_titles:
            if any(b in t for b in NON_TECH):
                role_score = 0.25 if ai_count >= 3 else 0.08
                role_tier = "adjacent" if ai_count >= 3 else "non_technical"
                break

    if role_tier == "unknown":
        ai_count = sum(1 for kw in AI_WORDS if kw in summary or kw in headline)
        role_score = min(0.65, 0.2 + ai_count * 0.1)
        role_tier = "moderate_tech_ai" if ai_count >= 4 else "adjacent"

    # ── 2. Skills Score ──────────────────────────────────────────────────────
    group_scores = {}
    matched_groups = set()
    sk_assess = sig.get("skill_assessment_scores", {})

    for sk in skills:
        nm  = sk["name"].lower().strip()
        wt  = SKILL_MAP.get(nm)
        if wt is None:
            continue
        prof = PROF_W.get(sk.get("proficiency","intermediate"), 0.5)
        end  = min(0.3, math.log1p(sk.get("endorsements",0)) / 15)
        dur  = min(0.2, sk.get("duration_months",0) / 60)
        ass  = (sk_assess.get(sk["name"],0) / 100) * 0.2
        raw  = prof * (1 + end + dur + ass)
        if wt > group_scores.get(wt, 0):
            group_scores[wt] = max(group_scores.get(wt,0), min(1.0, raw))
        matched_groups.add(nm)

    # weighted sum across weight tiers
    skill_score = 0.0
    for wt, best in group_scores.items():
        skill_score += wt * best
    skill_score = min(1.0, skill_score / MAX_SKILL_WEIGHT)

    # ── 3. Career Quality Score ──────────────────────────────────────────────
    # Years sweet spot
    if   6  <= yoe <= 8:  yoe_s = 1.00
    elif 5  <= yoe <  6:  yoe_s = 0.85
    elif 8  < yoe <= 9:   yoe_s = 0.85
    elif 4  <= yoe <  5:  yoe_s = 0.65
    elif 9  < yoe <= 12:  yoe_s = 0.65
    elif 3  <= yoe <  4:  yoe_s = 0.45
    else:                 yoe_s = 0.25

    cons_ratio  = (consulting_m / total_m) if total_m else 0.5
    company_s   = 1.0 - cons_ratio * 0.7
    prod_s      = min(1.0, prod_hits / 16)
    ai_prog     = min(1.0, (ai_ml_m / max(total_m,1)) * 2)
    research_p  = 0.5 if is_pure_research else 1.0
    career_score = (0.30*yoe_s + 0.25*company_s + 0.25*prod_s + 0.20*ai_prog) * research_p

    # ── 4. Semantic Fit Score (fast set ops) ─────────────────────────────────
    # Tokenize full text into words (one-time split)
    words_set = set(full_text.split())
    pos_count = len(JD_WORDS & words_set)
    pos_score = min(1.0, pos_count / 12)

    # Bigram check (only on substring match, much smaller set)
    bg_count = sum(1 for bg in JD_BIGRAMS if bg in full_text)
    bg_bonus = min(0.3, bg_count / 8)

    neg_count = sum(1 for nw in NEG_WORDS if nw in full_text)
    neg_pen   = min(0.4, neg_count * 0.08)

    semantic_score = max(0.0, pos_score + bg_bonus - neg_pen)

    # ── 5. Behavioral Score ──────────────────────────────────────────────────
    la = sig.get("last_active_date","")
    try:
        ld = datetime.strptime(la[:10], "%Y-%m-%d").date()
        di = (TODAY - ld).days
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

    # ── 6. Location Score ────────────────────────────────────────────────────
    reloc  = sig.get("willing_to_relocate", False)
    notice = sig.get("notice_period_days", 60)
    mode   = sig.get("preferred_work_mode","hybrid")

    if country == "india":
        city_s = 1.0 if any(ci in location for ci in INDIA_CITIES) else 0.8
    else:
        city_s = 0.35 if reloc else 0.15

    not_s = 1.0 if notice<=15 else 0.90 if notice<=30 else 0.70 if notice<=45 \
            else 0.55 if notice<=60 else 0.40 if notice<=90 else 0.25
    mode_s = 1.0 if mode in ("hybrid","flexible") else 0.75
    location_score = 0.55*city_s + 0.35*not_s + 0.10*mode_s

    # ── Weighted composite ───────────────────────────────────────────────────
    composite = (0.20*role_score + 0.30*skill_score + 0.20*career_score +
                 0.15*semantic_score + 0.10*behavioral_score + 0.05*location_score)

    # ── Red-flag multipliers ─────────────────────────────────────────────────
    flags = []
    mult = 1.0

    if role_tier == "non_technical":
        mult *= 0.05; flags.append("non_technical")

    all_cons = len(company_names) > 0 and all(
        any(cg in cn for cg in CONSULTING) for cn in company_names
    )
    if all_cons:
        mult *= 0.35; flags.append("pure_consulting")

    if country != "india" and not reloc:
        mult *= 0.25; flags.append("outside_india_no_reloc")

    # Only-recent-LLM penalty: career text has langchain/chatgpt but no pre-LLM AI
    recent_llm = ("langchain" in full_career or "chatgpt" in full_career)
    pre_llm_ai = any(w in full_career for w in
                     ("recommendation","retrieval","ranking","embedding","machine learning"))
    if recent_llm and not pre_llm_ai:
        mult *= 0.5; flags.append("only_recent_llm")

    final = composite * mult
    return final, role_tier, matched_groups, flags, role_score, skill_score, career_score, semantic_score, behavioral_score, location_score


# ── Reasoning generator ──────────────────────────────────────────────────────

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
    rr     = sig.get("recruiter_response_rate",0)
    notice = sig.get("notice_period_days",60)
    open_w = sig.get("open_to_work_flag", False)
    la     = sig.get("last_active_date","")

    if "non_technical" in flags:
        return (f"{title} ({yoe:.0f}y, {co}) — role domain mismatches JD; "
                f"AI skills listed but career trajectory is non-technical.")

    if "pure_consulting" in flags:
        return (f"{title} with {yoe:.1f}y, but entire career at IT services firms; "
                f"JD explicitly values product-company background.")

    if "outside_india_no_reloc" in flags:
        return (f"{title} in {loc} with no relocation intent; "
                f"role is India-based with no visa sponsorship.")

    if role_tier == "honeypot":
        return "Profile contains inconsistent timeline data — excluded as likely honeypot."

    # Build positive signals
    core_matched = [_GROUP_DISPLAY.get(m, m) for m in matched
                    if m in {"pinecone","weaviate","qdrant","milvus","faiss","opensearch",
                              "elasticsearch","embeddings","sentence-transformers","ranking",
                              "retrieval","recommendation systems"}][:3]
    parts = []
    if role_tier == "strong_ai":
        parts.append(f"{title} with {yoe:.1f}y AI/ML experience at {co}")
    else:
        parts.append(f"{title} ({yoe:.1f}y) at {co}")

    if core_matched:
        parts.append(f"covers {', '.join(core_matched)}")

    if t_s >= 0.65:
        parts.append("strong retrieval/ranking stack alignment")
    elif t_s >= 0.35:
        parts.append("partial technical alignment")

    # Behavioral flags
    concerns = []
    if not open_w:
        concerns.append("not open-to-work")
    if rr < 0.15:
        concerns.append(f"low response rate ({rr:.0%})")
    if notice > 90:
        concerns.append(f"notice {notice}d")
    elif notice <= 30:
        parts.append(f"notice ≤{notice}d (ideal)")

    s1 = f"{parts[0]}; {'; '.join(parts[1:3])}."
    s2 = f" Concerns: {', '.join(concerns)}." if concerns else ""
    return (s1 + s2).strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(candidates_path: str, output_path: str):
    import time
    t0 = time.time()

    TOP_K = 300  # keep top-300 for safety, take top-100 for output
    heap  = []   # min-heap of (final_score, cid)
    heap_full = {}  # cid → (result_tuple, candidate_dict) for top-300
    n = 0

    opener = open
    if candidates_path.endswith(".gz"):
        import gzip
        opener = lambda p, **kw: gzip.open(p, "rt", **kw)
        read_bytes = False
    else:
        read_bytes = True

    if read_bytes:
        f = open(candidates_path, "rb")
    else:
        f = opener(candidates_path, encoding="utf-8")

    for line in f:
        line = line.strip()
        if not line:
            continue
        c = _loads(line)
        result = _score(c)
        final = result[0]
        cid   = c["candidate_id"]
        n += 1

        if len(heap) < TOP_K:
            heapq.heappush(heap, (final, cid))
            heap_full[cid] = (result, c)
        elif final > heap[0][0]:
            _old_score, old_cid = heapq.heapreplace(heap, (final, cid))
            heap_full.pop(old_cid, None)
            heap_full[cid] = (result, c)

    f.close()
    t1 = time.time()
    print(f"Scored {n:,} candidates in {t1-t0:.1f}s", file=sys.stderr)

    # Sort top-300 descending, take top-100
    top = sorted(heap, key=lambda x: -x[0])[:100]

    max_s = top[0][0]; min_s = top[-1][0]
    rng   = max_s - min_s if max_s > min_s else 1.0

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["candidate_id","rank","score","reasoning"])
        for rank_i, (score, cid) in enumerate(top, 1):
            result, candidate = heap_full[cid]
            norm = round(0.400 + (score - min_s) / rng * 0.595, 4)
            reason = _reasoning(candidate, result)
            w.writerow([cid, rank_i, norm, reason])

    t2 = time.time()
    print(f"Output written to {output_path} (total: {t2-t0:.1f}s)", file=sys.stderr)

    # Summary
    top10 = top[:10]
    print(f"\nTop-10 candidates:", file=sys.stderr)
    for i, (s, cid) in enumerate(top10, 1):
        result, cand = heap_full[cid]
        tier = result[1]
        print(f"  {i:2d}. {cid}  score={s:.4f}  tier={tier}  "
              f"title={cand['profile']['current_title'][:35]}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.candidates, args.out)
