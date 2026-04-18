# D:\Project\DBSCAN\gemini_helper.py
import streamlit as st
import pandas as pd
import json
import asyncio
import requests
import random
from typing import List, Dict, Optional

# ====================== GEMINI HELPER (Robust & Production Ready) ======================
MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

def safe_json_extract(text: str):
    """Robust JSON extraction that handles code blocks and malformed responses."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1:
                start = text.find('[')
                end = text.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except:
            pass
    return None


async def call_gemini(prompt: str, api_key: str, max_retries: int = 3):
    """Single robust Gemini call with retry + model fallback + 503 handling."""
    if not api_key:
        return None

    for attempt in range(max_retries):
        for model in MODEL_FALLBACKS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
            }
            try:
                response = await asyncio.to_thread(
                    lambda: requests.post(url, json=payload, timeout=45).json()
                )

                if isinstance(response, dict) and "error" in response:
                    err = response["error"]
                    if err.get("code") == 503:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        st.toast(f"🔄 Gemini {model} overloaded (503). Retrying in {sleep_time:.1f}s...", icon="⏳")
                        await asyncio.sleep(sleep_time)
                        continue

                candidates = response.get("candidates")
                if not candidates:
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    continue

                text = parts[0].get("text", "")
                parsed = safe_json_extract(text)
                if parsed:
                    st.toast(f"✅ Gemini succeeded with {model}", icon="🚀")
                    return parsed

            except Exception:
                continue

        await asyncio.sleep(2 ** attempt)

    return None


# ====================== CLUSTER NAMES (ONE API CALL ONLY) ======================
@st.cache_data(show_spinner="Generating smart cluster names...", ttl=3600)
def get_cluster_names_api(cluster_summary: pd.DataFrame, api_key: str) -> pd.DataFrame:
    if cluster_summary.empty or not api_key:
        return pd.DataFrame()

    noise_note = "\nNote: If cluster_id = -1, it is NOISE/OUTLIERS. Name it 'Outliers' or 'Noise Customers'." \
                 if -1 in cluster_summary.get("Cluster", pd.Series()).values else ""

    prompt = f"""You are a senior marketing analyst.
Return ONLY a valid JSON array.
Each object must have: cluster_id (int), name (2-4 words), description (1-2 short sentences).

{noise_note}

Data:
{cluster_summary.to_json(orient="records")}

Return ONLY the JSON array."""

    result = asyncio.run(call_gemini(prompt, api_key))
    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df.columns = [c.lower().strip() for c in df.columns]
    if "cluster_id" not in df.columns and "id" in df.columns:
        df.rename(columns={"id": "cluster_id"}, inplace=True)
    return df


def get_cluster_names_fallback(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Rule-based fallback when Gemini fails."""
    data = []
    global_rfm = cluster_summary["RFM Score"].median()
    global_inc = cluster_summary["Annual Income (k$)"].median()

    for _, row in cluster_summary.iterrows():
        cid = int(row.get("Cluster", row.get("cluster_id", 0)))
        rfm = row["RFM Score"]
        inc = row["Annual Income (k$)"]

        if cid == -1:
            name, desc = "Noise / Outliers", "Customers who do not fit any pattern – treat as high-risk or unique cases."
        elif rfm >= global_rfm and inc >= global_inc:
            name, desc = "Premium Loyalists", "High-value, high-income, very engaged customers."
        elif rfm >= global_rfm:
            name, desc = "Budget Loyalists", "Frequent buyers but price-sensitive."
        elif inc >= global_inc:
            name, desc = "High-Potential", "High income but low engagement – needs reactivation."
        else:
            name, desc = "Churn Risk", "Low engagement & low value – aggressive offers required."

        data.append({"cluster_id": cid, "name": name, "description": desc})

    return pd.DataFrame(data)


# ====================== EMAIL TEMPLATES (ONE API CALL ONLY) ======================
@st.cache_data(show_spinner="Generating all email templates...", ttl=1800)
def generate_all_email_templates(cluster_summary: pd.DataFrame, discount_range: str, api_key: str) -> List[Dict]:
    if cluster_summary.empty or not api_key:
        return []

    prompt = f"""You are an expert email marketing copywriter.
Generate professional emails for a {discount_range} discount offer.

Return ONLY a JSON array. Each object must have:
- cluster_id (integer)
- subject (string, max 60 chars)
- body (string, short professional email, 4-6 sentences)

Cluster data:
{cluster_summary.to_json(orient="records")}

Return ONLY the JSON array."""

    result = asyncio.run(call_gemini(prompt, api_key))
    if not result or not isinstance(result, list):
        return []
    return result


def generate_email_fallback(cluster_summary: pd.DataFrame, discount_range: str) -> List[Dict]:
    templates = []
    for _, row in cluster_summary.iterrows():
        cid = int(row.get("Cluster", row.get("cluster_id", 0)))
        name = row.get("name", f"Cluster {cid}")
        templates.append({
            "cluster_id": cid,
            "subject": f"Exclusive {discount_range} Offer Just for {name}!",
            "body": f"""Hi there,

We noticed you belong to our valued {name} segment.

As a special thank you, enjoy {discount_range} off your next purchase.

Use code THANKYOU{discount_range.replace('%','')} at checkout.

Best regards,
Your Customer Success Team"""
        })
    return templates