"""
Intelligent Candidate Ranker — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json, csv, io, re, math
from datetime import datetime

# ── Company prestige tiers ────────────────────────────────────────────────────
CO_TIER1 = frozenset(["google","meta","amazon","microsoft","apple","netflix","deepmind","openai",
                       "anthropic","nvidia","linkedin","uber","airbnb","stripe","databricks"])
CO_TIER2 = frozenset(["flipkart","zomato","swiggy","cred","meesho","razorpay","phonepe","paytm",
                       "ola","sarvam","adobe","atlassian","salesforce","twitter","bytedance",
                       "rephrase","freshworks","cleartax","zerodha","groww","nykaa","sharechat","juspay"])
CO_TIER3 = frozenset(["byju","unacademy","vedantu","snapdeal","indiamart","justdial","makemytrip","oyo"])
ELITE_EDU = frozenset(["iit","iim","bits","nit","iisc","iiser","iiit","stanford","mit","cmu",
                        "carnegie mellon","berkeley","oxford","cambridge","eth zurich","waterloo",
                        "toronto","montreal","nyu","columbia"])
_METRIC_RE = re.compile(
    r'\d+[%xX×]\s*(?:improvement|reduction|increase|faster|lift|gain)|'
    r'(?:million|billion|crore|lakh)\s*(?:users|requests|queries|impressions)|'
    r'p\d{2}\s*latency|\d+\s*ms\s*(?:latency|p99)|serving\s*\d+',
    re.IGNORECASE
)
_SENIORITY = {"intern":0,"junior":1,"associate":2,"mid":3,"senior":4,"lead":5,"staff":6,"principal":7,"director":8,"vp":9}

def _company_prestige(company_names):
    best = 0.4
    for cn in company_names:
        if any(t in cn for t in CO_TIER1): best = max(best, 1.00)
        elif any(t in cn for t in CO_TIER2): best = max(best, 0.85)
        elif any(t in cn for t in CO_TIER3): best = max(best, 0.70)
        else: best = max(best, 0.60)
    return best

def _education_score(edu):
    if not edu: return 0.5
    for e in edu:
        inst = (e.get("institution") or e.get("school") or e.get("degree") or "").lower()
        if any(u in inst for u in ELITE_EDU): return 1.0
    return 0.55

def _quantified_impact(career):
    hits = sum(1 for j in career if _METRIC_RE.search(j.get("description","") or ""))
    return min(1.0, hits / 3)

def _career_progression(career_titles):
    if len(career_titles) < 2: return 0.6
    levels = []
    for t in career_titles:
        for kw, lv in _SENIORITY.items():
            if kw in t: levels.append(lv); break
        else: levels.append(3)
    mid = len(levels) // 2
    early = sum(levels[:mid]) / max(mid, 1)
    late  = sum(levels[mid:]) / max(len(levels)-mid, 1)
    if late > early + 0.5: return 1.0
    if late < early - 0.5: return 0.3
    return 0.6

def _skill_recency(career, skill_map):
    if not career: return 1.0
    recent_text = " ".join((j.get("description","") or "").lower() for j in career[-2:])
    high_val = sum(1 for sk, wt in skill_map.items() if wt >= 2.0 and sk in recent_text)
    return min(1.0, 0.7 + high_val * 0.1)

# Optional speedup
try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

TODAY = datetime.now()

# ══════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════

DOMAINS = {
    "ml_ai": {
        "label": "ML / AI Engineer", "icon": "🤖",
        "keywords": ["machine learning","deep learning","nlp","neural network","recommendation","retrieval","ranking","embedding","vector","llm","language model","transformer","bert","gpt","fine-tuning","semantic search","pytorch","tensorflow","hugging face","data science"],
        "skills": {
            "embeddings":2.5,"sentence-transformers":2.5,"semantic search":2.5,"faiss":2.5,
            "pinecone":2.5,"weaviate":2.5,"qdrant":2.5,"milvus":2.5,"vector database":2.5,"ann":2.5,
            "ranking":2.0,"retrieval":2.0,"bm25":2.0,"learning to rank":2.0,"ndcg":2.0,
            "recommendation systems":2.0,"haystack":2.0,"llamaindex":2.0,
            "pytorch":1.8,"tensorflow":1.8,"hugging face":1.8,"transformers":1.8,
            "scikit-learn":1.5,"python":1.5,"numpy":1.2,"pandas":1.2,
            "xgboost":1.5,"lightgbm":1.5,"fine-tuning":1.2,"lora":1.2,"rlhf":1.2,
            "mlflow":1.0,"wandb":1.0,"langchain":0.9,
        },
        "strong_titles": ["ml engineer","machine learning engineer","ai engineer","research scientist","nlp engineer","data scientist","applied scientist","research engineer","recommendation engineer","search engineer","ranking engineer","deep learning engineer","llm engineer","senior ml","senior ai","senior data scientist","principal ml","staff ml"],
        "mod_titles": ["software engineer","backend engineer","data engineer","platform engineer","devops","sre"],
        "non_titles": ["hr manager","marketing","accountant","graphic designer","customer support","sales","java developer",".net developer","android","ios","qa engineer","frontend"],
        "consulting": ["tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra","mindtree","mphasis"],
    },
    "data_engineering": {
        "label": "Data Engineer", "icon": "🔧",
        "keywords": ["data pipeline","etl","spark","kafka","airflow","data warehouse","dbt","bigquery","snowflake","redshift","data lake","streaming","databricks"],
        "skills": {
            "apache spark":2.5,"kafka":2.5,"airflow":2.5,"dbt":2.5,"flink":2.5,
            "bigquery":2.0,"snowflake":2.0,"redshift":2.0,"databricks":2.0,"pyspark":2.0,
            "python":1.8,"sql":1.8,"scala":1.5,"java":1.2,
            "kubernetes":1.2,"docker":1.2,"terraform":1.0,"aws":1.0,"gcp":1.0,
        },
        "strong_titles": ["data engineer","senior data engineer","analytics engineer","data platform engineer","etl developer","lead data engineer","staff data engineer"],
        "mod_titles": ["software engineer","backend engineer","cloud engineer","platform engineer","sre"],
        "non_titles": ["marketing","accountant","hr","sales","qa","frontend","android"],
        "consulting": ["tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra"],
    },
    "backend": {
        "label": "Backend Engineer", "icon": "⚙️",
        "keywords": ["rest api","microservices","distributed systems","backend","server","node.js","spring boot","django","fastapi","golang","java","system design","high availability","low latency"],
        "skills": {
            "java":2.5,"spring boot":2.5,"go":2.5,"golang":2.5,"python":2.0,
            "node.js":2.0,"nodejs":2.0,"django":1.8,"fastapi":1.8,"flask":1.5,
            "microservices":2.0,"rest api":2.0,"grpc":2.0,"kafka":1.5,
            "redis":1.5,"postgresql":1.5,"mysql":1.5,"mongodb":1.5,
            "kubernetes":1.8,"docker":1.5,"aws":1.5,"gcp":1.2,"azure":1.2,
        },
        "strong_titles": ["backend engineer","software engineer","senior software engineer","staff engineer","principal engineer","api engineer","senior backend","lead backend"],
        "mod_titles": ["fullstack","devops","sre","data engineer","cloud engineer"],
        "non_titles": ["marketing","hr","sales","accountant","android","ios","qa","frontend"],
        "consulting": ["tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra"],
    },
    "frontend": {
        "label": "Frontend Engineer", "icon": "🎨",
        "keywords": ["react","angular","vue","frontend","javascript","typescript","css","html","web app","responsive","accessibility","web performance"],
        "skills": {
            "react":2.5,"next.js":2.5,"typescript":2.5,"javascript":2.0,"vue":2.0,"angular":2.0,
            "css":1.5,"html":1.2,"tailwind":1.5,"webpack":1.2,"vite":1.2,
            "graphql":1.5,"jest":1.2,"cypress":1.2,"playwright":1.2,"accessibility":1.5,
        },
        "strong_titles": ["frontend engineer","ui engineer","javascript developer","react developer","web developer","senior frontend","staff frontend"],
        "mod_titles": ["fullstack","software engineer","mobile developer"],
        "non_titles": ["backend","data engineer","ml engineer","marketing","hr","sales"],
        "consulting": ["tcs","infosys","wipro","accenture","cognizant"],
    },
    "devops": {
        "label": "DevOps / SRE", "icon": "☁️",
        "keywords": ["kubernetes","docker","terraform","ci/cd","devops","sre","cloud","aws","gcp","azure","infrastructure","monitoring","observability","helm","gitops"],
        "skills": {
            "kubernetes":2.5,"helm":2.0,"terraform":2.5,"ansible":2.0,"docker":2.0,
            "aws":2.0,"gcp":2.0,"azure":2.0,"ci/cd":2.0,"github actions":1.8,"argocd":2.0,
            "prometheus":1.8,"grafana":1.5,"datadog":1.5,"opentelemetry":1.8,
            "python":1.2,"go":1.5,"bash":1.5,"linux":1.5,
        },
        "strong_titles": ["devops engineer","sre","site reliability engineer","platform engineer","cloud engineer","infrastructure engineer","staff sre","lead sre"],
        "mod_titles": ["backend engineer","software engineer","data engineer"],
        "non_titles": ["marketing","hr","sales","accountant","frontend","android"],
        "consulting": ["tcs","infosys","wipro","accenture","cognizant"],
    },
    "data_analyst": {
        "label": "Data Analyst", "icon": "📊",
        "keywords": ["sql","tableau","power bi","looker","analytics","reporting","dashboard","business intelligence","metrics","a/b test","statistics","experimentation"],
        "skills": {
            "sql":2.5,"python":2.0,"tableau":2.5,"power bi":2.5,"looker":2.0,
            "excel":1.8,"statistics":2.0,"a/b testing":2.0,"experimentation":2.0,
            "pandas":1.5,"numpy":1.2,"dbt":1.5,"bigquery":1.5,"snowflake":1.2,
        },
        "strong_titles": ["data analyst","senior data analyst","business analyst","bi analyst","analytics engineer","product analyst","growth analyst"],
        "mod_titles": ["data scientist","product manager","software engineer"],
        "non_titles": ["hr","accountant","sales","marketing","graphic designer","android","ios"],
        "consulting": ["tcs","infosys","wipro","accenture"],
    },
}

STOPWORDS = set("a an the is are was were be been being have has had do does did will would could should may might must shall can i we you he she it they them their this that these those of in on at by for with from to into through during before after above below up down out off over under again further then once here there when where why how all both each few more most no nor not only own same so than too very just now".split())
PROF_W = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}


# ══════════════════════════════════════════════════════════════
# JD PARSER
# ══════════════════════════════════════════════════════════════

def parse_jd(text: str) -> dict:
    lower = text.lower()

    best_domain, best_score = "ml_ai", 0
    for key, dom in DOMAINS.items():
        score = sum(1 for k in dom["keywords"] if k in lower)
        if score > best_score:
            best_score, best_domain = score, key

    dom = DOMAINS[best_domain]

    skill_map = {sk: wt for sk, wt in dom["skills"].items() if sk in lower}
    if len(skill_map) < 3:
        skill_map = dict(sorted(dom["skills"].items(), key=lambda x: -x[1])[:8])

    words = re.findall(r'\b[a-z]{4,}\b', lower)
    freq: dict = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    jd_words = set(w for w, f in freq.items() if f >= 2)
    for sk in skill_map:
        for tok in sk.split():
            jd_words.add(tok)

    m = re.search(r'(\d+)\s*(?:to|-|–)\s*(\d+)\s*years?', text, re.I) or \
        re.search(r'(\d+)\+\s*years?', text, re.I)
    min_yoe, max_yoe = 0, 99
    if m:
        min_yoe = int(m.group(1))
        max_yoe = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else min_yoe + 4

    india_words = ["india","bangalore","bengaluru","pune","hyderabad","noida","mumbai","delhi","gurgaon","chennai"]
    is_india = any(w in lower for w in india_words)
    preferred_cities = {c for c in ["bangalore","bengaluru","pune","hyderabad","noida","mumbai","delhi","gurgaon","gurugram","chennai","kolkata","ahmedabad","greater noida"] if c in lower}
    anti_consulting = any(p in lower for p in ["product company","product-company","startup","not from consulting"])

    top_wts = sorted(skill_map.values(), reverse=True)
    max_skill_w = max((top_wts[0] if top_wts else 2.5) * 3 + sum(top_wts[3:5]), 5.0)

    return {
        "domain_key": best_domain,
        "domain": dom,
        "skill_map": skill_map,
        "jd_words": jd_words,
        "min_yoe": min_yoe,
        "max_yoe": max_yoe,
        "is_india": is_india,
        "preferred_cities": preferred_cities,
        "anti_consulting": anti_consulting,
        "max_skill_w": max_skill_w,
    }


# ══════════════════════════════════════════════════════════════
# SCORER
# ══════════════════════════════════════════════════════════════

def detect_honeypot(c: dict) -> bool:
    yoe = (c.get("profile") or {}).get("years_of_experience") or 0
    cur_year = TODAY.year
    for e in c.get("education") or []:
        ey = e.get("end_year")
        if not ey:
            continue
        if ey > cur_year or ey < 1960:
            return True
        if yoe > (cur_year - ey + 2):
            return True
    total_m = sum((j.get("duration_months") or 0) for j in (c.get("career_history") or []))
    if total_m > 0 and total_m > (yoe + 5) * 12:
        return True
    for j in c.get("career_history") or []:
        sd = j.get("start_date", "")
        if not sd:
            continue
        try:
            sy = int(str(sd)[:4])
            if sy > cur_year or sy < 1970:
                return True
        except Exception:
            pass
    return False


def score_candidate(c: dict, weights: dict, cfg: dict) -> dict:
    ZERO = {"final": 0.0, "tier": "honeypot", "tier_label": "Honeypot",
            "matched": [], "flags": ["honeypot"],
            "subs": {k: 0.0 for k in ["role","tech","career","semantic","behav","loc"]}}
    if detect_honeypot(c):
        return ZERO

    dom = cfg["domain"]
    profile  = c.get("profile") or {}
    career   = c.get("career_history") or []
    skills   = c.get("skills") or []
    sig      = c.get("redrob_signals") or {}

    title    = (profile.get("current_title") or "").lower()
    summary  = (profile.get("summary") or "").lower()
    headline = (profile.get("headline") or "").lower()
    yoe      = profile.get("years_of_experience") or 0
    country  = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()

    career_texts, company_names, career_titles = [], [], []
    consulting_m = total_m = prod_hits = prod_m = 0
    is_pure_research = len(career) > 0
    PROD_KW = {"production","deployed","shipped","at scale","real users","live","latency","throughput"}

    for j in career:
        jt  = (j.get("title") or "").lower()
        jco = (j.get("company") or "").lower().strip()
        jd  = (j.get("description") or "").lower()
        m   = j.get("duration_months") or 0
        career_texts.append(jd); career_titles.append(jt); company_names.append(jco); total_m += m
        if any(cg in jco for cg in dom["consulting"]):
            consulting_m += m
        prod_hits = min(prod_hits + sum(1 for kw in PROD_KW if kw in jd), 16)
        if any(a in jt for a in dom["strong_titles"]):
            prod_m += m
        if not any(r in jt for r in ["research scientist","research engineer","phd","professor","academic"]):
            is_pure_research = False

    full_career  = " ".join(career_texts)
    skills_text  = " ".join((s.get("name") or "").lower() for s in skills)
    all_titles   = [title] + career_titles
    full_text    = f"{summary} {headline} {full_career} {skills_text}"

    # ── Role legitimacy
    role_tier, role_score = "unknown", 0.0
    for t in all_titles:
        if any(a in t for a in dom["strong_titles"]):
            role_score, role_tier = 1.0, "strong"; break
    if role_tier == "unknown":
        dom_hits = sum(1 for k in dom["keywords"] if k in summary or k in headline)
        for t in all_titles:
            if any(a in t for a in dom["mod_titles"]):
                role_score = 0.75 if dom_hits >= 2 else 0.5; role_tier = "moderate"; break
    if role_tier == "unknown":
        dom_hits = sum(1 for k in dom["keywords"] if k in summary or k in headline)
        for t in all_titles:
            if any(a in t for a in dom["non_titles"]):
                role_score = 0.25 if dom_hits >= 3 else 0.06
                role_tier  = "adjacent" if dom_hits >= 3 else "excluded"; break
    if role_tier == "unknown":
        dom_hits = sum(1 for k in dom["keywords"] if k in summary or k in headline)
        role_score = min(0.6, 0.15 + dom_hits * 0.1)
        role_tier  = "moderate" if dom_hits >= 4 else "adjacent"

    # ── Skill score
    matched_skills, group_best = [], {}
    sk_assess = sig.get("skill_assessment_scores") or {}
    for sk in skills:
        nm = (sk.get("name") or "").lower().strip()
        wt = cfg["skill_map"].get(nm)
        if wt is None:
            continue
        prof = PROF_W.get(sk.get("proficiency"), 0.5)
        end  = min(0.3, math.log1p(sk.get("endorsements") or 0) / 15)
        dur  = min(0.2, (sk.get("duration_months") or 0) / 60)
        ass  = ((sk_assess.get(sk.get("name", "")) or 0) / 100) * 0.2
        raw  = min(1.0, prof * (1 + end + dur + ass))
        group_best[wt] = max(group_best.get(wt, 0.0), raw)
        matched_skills.append(sk.get("name") or nm)
    skill_score = min(1.0, sum(wt * best for wt, best in group_best.items()) / cfg["max_skill_w"])
    # Skill recency boost
    skill_score = min(1.0, skill_score * _skill_recency(career, cfg["skill_map"]))

    # ── Career quality (enhanced)
    edu        = c.get("education") or []
    yoe_ideal  = (cfg["min_yoe"] or 4) + round(((cfg["max_yoe"] if cfg["max_yoe"] < 99 else 8) - (cfg["min_yoe"] or 4)) * 0.5)
    yoe_s      = (1.0 if abs(yoe - yoe_ideal) <= 1
                  else 0.75 if yoe >= (cfg["min_yoe"] or 4)
                  else 0.5  if yoe >= (cfg["min_yoe"] or 4) - 1
                  else 0.3)
    company_s   = 1.0 - (consulting_m / total_m * 0.6 if total_m else 0.3)
    prestige_s  = _company_prestige(company_names)
    prod_s      = min(1.0, prod_hits / 12)
    impact_s    = _quantified_impact(career)
    progression_s= _career_progression(career_titles)
    edu_s       = _education_score(edu)
    domain_m_   = min(1.0, (prod_m / max(total_m, 1)) * 2)
    career_score = (
        0.20 * yoe_s + 0.15 * company_s + 0.15 * prestige_s +
        0.15 * prod_s + 0.10 * impact_s + 0.10 * progression_s +
        0.10 * edu_s + 0.05 * domain_m_
    ) * (0.55 if is_pure_research else 1.0)

    # ── Semantic fit
    words_set = set(full_text.split())
    pos_count = sum(1 for w in cfg["jd_words"] if w in words_set)
    semantic_score = min(1.0, pos_count / max(len(cfg["jd_words"]) * 0.25, 8))

    # ── Behavioral
    la  = sig.get("last_active_date") or ""
    rec = 0.4
    if la:
        try:
            di = (TODAY - datetime.fromisoformat(str(la)[:10])).days
            rec = 1.0 if di<=30 else 0.8 if di<=90 else 0.5 if di<=180 else 0.3 if di<=365 else 0.1
        except Exception:
            pass
    open_w  = 1.0 if sig.get("open_to_work_flag") else 0.4
    rr      = sig.get("recruiter_response_rate") or 0.5
    arh     = sig.get("avg_response_time_hours") or 48
    rt_s    = 1.0 if arh<=4 else 0.85 if arh<=24 else 0.65 if arh<=72 else 0.45 if arh<=168 else 0.2
    saved   = sig.get("saved_by_recruiters_30d") or 0
    views   = sig.get("profile_views_received_30d") or 0
    mkt_s   = min(1.0, (math.log1p(saved)/4 + math.log1p(views)/6) / 2 * 1.5)
    icr     = sig.get("interview_completion_rate") or 0.7
    comp    = (sig.get("profile_completeness_score") or 70) / 100
    ver     = ((1 if sig.get("verified_email") else 0) +
               (1 if sig.get("verified_phone") else 0) +
               (1 if sig.get("linkedin_connected") else 0)) / 3
    gh      = max(0, sig.get("github_activity_score") or 0) / 100
    behav_score = (0.20*rec + 0.15*open_w + 0.20*rr + 0.10*rt_s +
                   0.15*mkt_s + 0.10*icr + 0.10*(comp*0.4 + ver*0.4 + gh*0.2))

    # ── Location
    reloc  = sig.get("willing_to_relocate") or False
    notice = sig.get("notice_period_days") or 60
    mode   = sig.get("preferred_work_mode") or "hybrid"
    if cfg["is_india"]:
        pref   = cfg["preferred_cities"]
        city_s = (1.0 if country == "india" and (not pref or any(ci in location for ci in pref))
                  else 0.75 if country == "india"
                  else 0.35 if reloc else 0.1)
    else:
        city_s = 0.8
    not_s   = (1.0 if notice<=15 else 0.9 if notice<=30 else 0.7 if notice<=45
               else 0.55 if notice<=60 else 0.4 if notice<=90 else 0.25)
    loc_score = 0.55*city_s + 0.35*not_s + 0.10*(1.0 if mode in ("hybrid","flexible") else 0.75)

    subs = {"role": role_score, "tech": skill_score, "career": career_score,
            "semantic": semantic_score, "behav": behav_score, "loc": loc_score}
    composite = sum(weights[k] * v for k, v in subs.items())

    flags, mult = [], 1.0
    if role_tier == "excluded":
        mult *= 0.05; flags.append("role_mismatch")
    if (cfg["anti_consulting"] and company_names
            and all(any(cg in cn for cg in dom["consulting"]) for cn in company_names if cn)):
        mult *= 0.35; flags.append("pure_consulting")
    if cfg["is_india"] and country != "india" and not reloc:
        mult *= 0.25; flags.append("outside_preferred_location")

    tier_labels = {"strong":"Strong match","moderate":"Moderate","adjacent":"Adjacent",
                   "excluded":"Excluded","honeypot":"Honeypot"}
    return {
        "final":      composite * mult,
        "tier":       role_tier,
        "tier_label": tier_labels.get(role_tier, role_tier),
        "matched":    matched_skills,
        "flags":      flags,
        "subs":       subs,
    }


# ══════════════════════════════════════════════════════════════
# FILE LOADERS
# ══════════════════════════════════════════════════════════════

def load_jsonl(content: bytes) -> list:
    rows = []
    for line in content.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(loads(line))
            except Exception:
                pass
    return rows

def load_json(content: bytes) -> list:
    data = loads(content)
    return data if isinstance(data, list) else [data]

COL_ALIASES = {
    "name":            ["name","full_name","anonymized_name","candidate_name","fullname","first_name"],
    "id":              ["candidate_id","id","cand_id","applicant_id"],
    "title":           ["current_title","title","role","position","job_title","designation"],
    "yoe":             ["years_of_experience","yoe","experience_years","experience","total_experience"],
    "company":         ["current_company","company","employer","organization"],
    "location":        ["location","city","current_location"],
    "country":         ["country","country_code"],
    "skills":          ["skills","skill_list","technical_skills","key_skills"],
    "summary":         ["summary","bio","about","profile_summary","description"],
    "open_to_work":    ["open_to_work","open_to_work_flag","available","actively_looking"],
    "response_rate":   ["response_rate","recruiter_response_rate","rr"],
    "notice_period":   ["notice_period_days","notice_period","notice"],
    "github_score":    ["github_activity_score","github_score","github"],
    "last_active":     ["last_active_date","last_active","last_login"],
    "interview_rate":  ["interview_completion_rate","interview_rate"],
    "willing_relocate":["willing_to_relocate","relocate","relocation"],
    "work_mode":       ["preferred_work_mode","work_mode","work_type"],
}

def detect_col_map(headers: list) -> dict:
    h_lower = [h.lower().strip().replace(" ","_") for h in headers]
    col_map = {}
    for field, aliases in COL_ALIASES.items():
        for alias in aliases:
            key = alias.replace(" ","_")
            if key in h_lower:
                col_map[field] = headers[h_lower.index(key)]; break
    return col_map

def load_csv(content: bytes) -> list:
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    headers = list(reader.fieldnames or [])
    col_map = detect_col_map(headers)
    rows = []
    for i, row in enumerate(reader):
        g  = lambda f: (row.get(col_map.get(f,""),"") or "").strip()
        gn = lambda f, d=0: (float(g(f)) if re.match(r'^-?\d+\.?\d*$', g(f)) else d)
        gb = lambda f: g(f).lower() in ("true","yes","1","y")
        yoe        = gn("yoe") or gn("experience")
        skills_raw = g("skills")
        skill_objs = [{"name":s.strip(),"proficiency":"intermediate","endorsements":0,"duration_months":0}
                      for s in re.split(r"[;,|]+", skills_raw) if s.strip()] if skills_raw else []
        cid  = g("id") or f"CAND_{i:07d}"
        name = g("name")
        rows.append({
            "candidate_id": cid,
            "profile": {
                "name": name, "anonymized_name": name,
                "current_title": g("title"),
                "years_of_experience": yoe,
                "current_company": g("company"),
                "location": g("location"),
                "country": g("country").lower() or "india",
                "summary": g("summary"),
                "headline": g("title"),
            },
            "career_history": [{"title":g("title"),"company":g("company"),
                                 "duration_months":round(yoe*12),
                                 "description":g("summary"),"industry":""}],
            "education": [],
            "skills": skill_objs,
            "redrob_signals": {
                "open_to_work_flag":          gb("open_to_work"),
                "recruiter_response_rate":     gn("response_rate", 0.6),
                "notice_period_days":          gn("notice_period", 30),
                "github_activity_score":       gn("github_score", -1),
                "last_active_date":            g("last_active"),
                "interview_completion_rate":   gn("interview_rate", 0.7),
                "profile_completeness_score":  gn("profile_complete", 70),
                "willing_to_relocate":         gb("willing_relocate"),
                "preferred_work_mode":         g("work_mode") or "hybrid",
                "avg_response_time_hours":     24,
                "saved_by_recruiters_30d":     0,
                "profile_views_received_30d":  0,
                "skill_assessment_scores":     {},
            },
        })
    return rows


# ══════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Intelligent Candidate Ranker", page_icon="🔍", layout="wide")

st.title("🔍 Intelligent Candidate Ranker")
st.caption("Multi-signal ranking engine · any role & JD · CSV / JSON / JSONL · no API calls")

# Session state init
for k, v in [("results", None), ("jd_config", None), ("candidates", None), ("file_name","")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # JD
    st.subheader("1 · Job Description")
    jd_file = st.file_uploader("Upload .txt / .md", type=["txt","md"], key="jd_file_up")
    jd_default = jd_file.read().decode("utf-8", errors="replace") if jd_file else ""
    jd_text = st.text_area("or paste JD here", value=jd_default, height=180,
                            placeholder="Paste job title, responsibilities, required skills, location…")

    if jd_text and len(jd_text) > 30:
        cfg = parse_jd(jd_text)
        st.session_state.jd_config = cfg
        dom = cfg["domain"]
        st.success(f"{dom['icon']} **{dom['label']}** detected")
        if cfg["min_yoe"] > 0:
            mx = cfg["max_yoe"] if cfg["max_yoe"] < 99 else "∞"
            st.caption(f"YoE target: {cfg['min_yoe']}–{mx} years")
        top_skills = ", ".join(s for s,_ in sorted(cfg["skill_map"].items(), key=lambda x:-x[1])[:6])
        st.caption(f"Key skills: {top_skills}")
    else:
        st.session_state.jd_config = None

    st.divider()

    # Weights
    st.subheader("2 · Scoring Weights")
    c1, c2 = st.columns(2)
    with c1:
        w_role   = st.slider("Role",      0, 70, 20)
        w_career = st.slider("Career",    0, 70, 25)
        w_behav  = st.slider("Behavioral",0, 70, 10)
    with c2:
        w_tech   = st.slider("Skills",    0, 70, 28)
        w_sem    = st.slider("Semantic",  0, 70, 12)
        w_loc    = st.slider("Location",  0, 70,  5)

    total_w = w_role + w_tech + w_career + w_sem + w_behav + w_loc
    (st.success if total_w == 100 else st.warning)(f"Total: {total_w}%" + (" ✓" if total_w == 100 else " — must be 100%"))
    weights = {k: v/100 for k,v in [("role",w_role),("tech",w_tech),("career",w_career),
                                      ("semantic",w_sem),("behav",w_behav),("loc",w_loc)]}


# ── TABS ─────────────────────────────────────────────────────
tab_upload, tab_results = st.tabs(["📂 Upload & Run", "📊 Results"])


# ── Tab 1: Upload & Run ──────────────────────────────────────
with tab_upload:
    st.subheader("Candidates File")
    st.caption("Supports **.jsonl** · **.json** (array) · **.csv** (any headers — auto-mapped)")

    load_method = st.radio("Load from", ["File uploader", "Local file path (for large files >200 MB)"], horizontal=True)

    def _parse_and_store(content: bytes, fname: str):
        with st.spinner("Parsing file…"):
            if fname.endswith(".csv"):
                candidates = load_csv(content)
                col_map = detect_col_map(list(csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace"))).fieldnames or []))
                st.success(f"CSV loaded · **{len(candidates):,}** candidates")
                with st.expander("Column mapping (auto-detected)"):
                    rows_cm = [[f, col_map.get(f, "—")] for f in COL_ALIASES]
                    st.dataframe(pd.DataFrame(rows_cm, columns=["Field","CSV column"]), hide_index=True, use_container_width=True)
            elif fname.endswith(".json"):
                candidates = load_json(content)
                st.success(f"JSON loaded · **{len(candidates):,}** candidates")
            else:
                candidates = load_jsonl(content)
                st.success(f"JSONL loaded · **{len(candidates):,}** candidates")
        st.session_state.candidates = candidates
        st.session_state.file_name  = fname
        st.session_state.results    = None

    if load_method == "File uploader":
        cand_file = st.file_uploader("Upload candidates", type=["jsonl","json","csv"])
        if cand_file and cand_file.name != st.session_state.file_name:
            _parse_and_store(cand_file.read(), cand_file.name)
    else:
        file_path = st.text_input("Full path to candidates file",
                                   placeholder=r"C:\path\to\candidates.jsonl")
        if st.button("Load from path") and file_path:
            import os
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
            elif file_path != st.session_state.file_name:
                with open(file_path, "rb") as f:
                    _parse_and_store(f.read(), os.path.basename(file_path))
                    st.session_state.file_name = file_path  # use full path as key

    st.divider()

    ready = (st.session_state.candidates is not None and
             st.session_state.jd_config  is not None and
             total_w == 100)

    if not st.session_state.jd_config:
        st.info("Paste a Job Description in the sidebar first.")
    elif st.session_state.candidates is None:
        st.info("Upload a candidates file above.")
    elif total_w != 100:
        st.warning("Adjust weights to sum to 100% before running.")

    if st.button("▶ Run Ranking", type="primary", disabled=not ready):
        cfg        = st.session_state.jd_config
        candidates = st.session_state.candidates
        n          = len(candidates)

        prog_bar = st.progress(0, text="Starting…")
        status   = st.empty()
        results  = []

        for i, c in enumerate(candidates):
            r    = score_candidate(c, weights, cfg)
            prof = c.get("profile") or {}
            sig  = c.get("redrob_signals") or {}
            name = (prof.get("name") or prof.get("anonymized_name") or
                    prof.get("full_name") or "").strip()

            results.append({
                "id":             c.get("candidate_id", f"CAND_{i:07d}"),
                "name":           name,
                "title":          prof.get("current_title",""),
                "yoe":            prof.get("years_of_experience") or 0,
                "company":        prof.get("current_company",""),
                "location":       prof.get("location",""),
                "score":          round(r["final"], 4),
                "tier":           r["tier"],
                "tier_label":     r["tier_label"],
                "open_to_work":   "Yes" if sig.get("open_to_work_flag") else "No",
                "notice_days":    sig.get("notice_period_days") or 60,
                "response_rate":  f"{(sig.get('recruiter_response_rate') or 0)*100:.0f}%",
                "matched_skills": ", ".join(r["matched"][:6]),
                "flags":          ", ".join(r["flags"]),
                "role_score":     round(r["subs"]["role"],     3),
                "tech_score":     round(r["subs"]["tech"],     3),
                "career_score":   round(r["subs"]["career"],   3),
                "semantic_score": round(r["subs"]["semantic"], 3),
                "behav_score":    round(r["subs"]["behav"],    3),
                "loc_score":      round(r["subs"]["loc"],      3),
                "_raw":           c,
            })

            if i % 500 == 0 or i == n - 1:
                pct = (i + 1) / n
                prog_bar.progress(pct, text=f"Scoring {i+1:,} / {n:,}…")
                status.caption(f"Processed {i+1:,} candidates")

        df = (pd.DataFrame(results)
                .sort_values("score", ascending=False)
                .reset_index(drop=True))
        df.insert(0, "rank", df.index + 1)
        st.session_state.results = df

        prog_bar.progress(1.0, text="Done!")
        status.success(f"Ranked **{len(df):,}** candidates. Open the **Results** tab.")


# ── Tab 2: Results ───────────────────────────────────────────
with tab_results:
    if st.session_state.results is None:
        st.info("Run the ranking first (Upload & Run tab).")
        st.stop()

    df = st.session_state.results

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total ranked",  f"{len(df):,}")
    c2.metric("Strong match",  f"{(df.tier=='strong').sum():,}")
    c3.metric("Red-flagged",   f"{(df['flags'] != '').sum():,}")
    c4.metric("Top-100 open",  f"{(df.head(100).open_to_work=='Yes').sum()}")

    st.divider()

    # Filters
    f1, f2, f3 = st.columns([3,2,2])
    q        = f1.text_input("Search name, title or company", "")
    tier_f   = f2.selectbox("Tier filter", ["All","strong","moderate","adjacent","excluded","honeypot"])
    otw_f    = f3.selectbox("Availability", ["All","Open to work only"])

    view = df.copy()
    if q:
        mask = (view.name.str.contains(q, case=False, na=False)   |
                view.title.str.contains(q, case=False, na=False)  |
                view.company.str.contains(q, case=False, na=False))
        view = view[mask]
    if tier_f != "All":
        view = view[view.tier == tier_f]
    if otw_f == "Open to work only":
        view = view[view.open_to_work == "Yes"]

    DISPLAY_COLS = ["rank","id","name","title","yoe","company","location",
                    "score","tier_label","open_to_work","notice_days",
                    "response_rate","matched_skills","flags"]

    st.dataframe(
        view[DISPLAY_COLS].head(300),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank":          st.column_config.NumberColumn("Rank",    width=55),
            "id":            st.column_config.TextColumn("ID",        width=110),
            "name":          st.column_config.TextColumn("Name",      width=130),
            "title":         st.column_config.TextColumn("Title"),
            "yoe":           st.column_config.NumberColumn("YoE",     width=55, format="%.1f"),
            "company":       st.column_config.TextColumn("Company",   width=140),
            "location":      st.column_config.TextColumn("Location",  width=110),
            "score":         st.column_config.NumberColumn("Score",   width=75, format="%.4f"),
            "tier_label":    st.column_config.TextColumn("Tier",      width=100),
            "open_to_work":  st.column_config.TextColumn("OTW",       width=50),
            "notice_days":   st.column_config.NumberColumn("Notice",  width=60),
            "response_rate": st.column_config.TextColumn("Resp rate", width=75),
            "matched_skills":st.column_config.TextColumn("Matched skills"),
            "flags":         st.column_config.TextColumn("Flags"),
        }
    )
    st.caption(f"Showing {min(300, len(view)):,} of {len(view):,} matches")

    # ── Candidate detail ──────────────────────────────────────
    st.divider()
    st.subheader("Candidate Detail")

    top_ids = view.head(50)["id"].tolist()
    if not top_ids:
        st.caption("No candidates match the current filter.")
    else:
        def fmt_option(cid):
            row = df[df.id == cid]
            if row.empty: return cid
            r = row.iloc[0]
            label = r["name"] or r["title"]
            return f"{cid} — {label} ({r['tier_label']})"

        sel = st.selectbox("Select candidate", top_ids, format_func=fmt_option)
        if sel:
            row = df[df.id == sel].iloc[0]
            d1, d2 = st.columns(2)

            with d1:
                if row["name"]:
                    st.markdown(f"### {row['name']}")
                    st.markdown(f"**{row['title']}**")
                else:
                    st.markdown(f"### {row['title']}")
                st.caption(f"{row['yoe']:.1f}y · {row['location']} · {row['company']}")
                st.markdown(f"**Score:** `{row['score']:.4f}`   **Tier:** {row['tier_label']}")
                if row["flags"]:
                    st.warning(f"⚠ {row['flags']}")

                st.markdown("**Score breakdown**")
                for k, lbl in [("role","Role legitimacy"),("tech","Technical skills"),
                                ("career","Career quality"),("semantic","Semantic fit"),
                                ("behav","Behavioral"),("loc","Location")]:
                    v = float(row[f"{k}_score"])
                    st.progress(v, text=f"{lbl}: {v*100:.1f}%")

            with d2:
                st.markdown("**Signals & logistics**")
                raw_sig = (row["_raw"].get("redrob_signals") or {})
                info = {
                    "Open to work":      row["open_to_work"],
                    "Response rate":     row["response_rate"],
                    "Notice period":     f"{row['notice_days']} days",
                    "Last active":       raw_sig.get("last_active_date","?"),
                    "Interview rate":    f"{(raw_sig.get('interview_completion_rate') or 0)*100:.0f}%",
                    "Work mode":         raw_sig.get("preferred_work_mode","?"),
                    "Willing to relocate":"Yes" if raw_sig.get("willing_to_relocate") else "No",
                    "GitHub score":      raw_sig.get("github_activity_score","Not linked"),
                    "Matched skills":    row["matched_skills"] or "—",
                }
                for k, v in info.items():
                    st.markdown(f"**{k}:** {v}")

    # ── Export ────────────────────────────────────────────────
    st.divider()
    top100 = df[df.tier != "honeypot"].head(100).copy()
    export_cols = ["id","rank","score","tier_label","name","title","yoe","company","matched_skills","flags"]
    top100_exp  = top100[export_cols].rename(columns={"id":"candidate_id","tier_label":"tier"})
    st.download_button(
        "⬇ Export top-100 CSV",
        data=top100_exp.to_csv(index=False).encode(),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )
