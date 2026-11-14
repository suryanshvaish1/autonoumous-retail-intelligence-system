# gemini_advisor.py - structured action plan + chat, grounded in computed analytics
import json
import re
import time
from google import genai
from google.genai import types

# Model names change over time. If you get "model not found", open Google AI Studio,
# check the current Flash model name, and edit this line.
MODEL_NAME = "gemini-3.5-flash"
# If the main model is overloaded (503), these are tried next. Edit names if Google changes them.
FALLBACK_MODELS = ["gemini-3.5-flash-lite"]
RETRIES_PER_MODEL = 3

GROUNDING_RULES = (
    "You are a retail business consultant for a store using camera analytics. "
    "You are given LIVE store data, 90-day HISTORICAL data, and outputs of a predictive "
    "model trained on SYNTHETIC data. Rules: use ONLY the numbers provided and never invent "
    "figures; clearly separate live data from historical data; model what-if results show "
    "what the model learned, not proven cause and effect, so recommend validating key actions "
    "with a small A/B test; items marked Basis=Assumption are assumptions, not model results; "
    "revenue figures are in Indian rupees (Rs). "
    "A DAILY TIME SERIES and a TWO-PERIOD COMPARISON may be provided; use them for "
    "questions about trends or comparing date ranges."
)

PLAIN_STYLE = (
    " WRITING STYLE: the reader is a busy shop owner with no data or statistics background. "
    "Use short sentences and everyday words. Avoid jargon (say 'time spent at the shelf' not "
    "'dwell', 'chance of buying' not 'conversion probability'; never mention MAPE, ROC-AUC, "
    "z-score or similar). Keep numbers to the few that matter, round them, and always say what "
    "a number MEANS in shop terms (e.g. 'about 3 fewer customers a day', not just '-5.2%'). "
    "Reasons for a change are likely explanations, not proven facts: word them as 'probably' "
    "or 'could be'. Always finish with concrete steps the owner can do in the shop."
)

PLAN_PROMPT = (
    GROUNDING_RULES
    + " Return ONLY a JSON array (no markdown, no commentary) of 3 to 5 objects, ordered by "
    "priority. Each object must have exactly these keys: "
    '"priority" (integer, 1 = most important), "title" (max 8 words), '
    '"action" (1-2 concrete sentences), '
    '"expected_impact" (use a Rs figure from the action table when relevant), '
    '"effort" ("Low", "Medium" or "High"), '
    '"evidence" (the specific numbers supporting it), '
    '"caveat" (one short sentence on the main uncertainty). '
    "Use the budget-optimised plan, alerts and trend comparison when they are relevant. "
    "Include one item that proposes an A/B test to validate the top recommendation, using "
    "the sample-size estimate if one is provided."
    + PLAIN_STYLE
    + " In the plan, write 'action' as a simple instruction a shop assistant could follow."
)

CHAT_PROMPT = (
    GROUNDING_RULES
    + PLAIN_STYLE
    + " Format every answer to a business question like this, using markdown:\n"
    "**In short:** one plain sentence with the main takeaway.\n"
    "**What changed:** 2-3 short bullets. Each bullet says what happened in simple words, "
    "with at most one or two key numbers and what they mean.\n"
    "**Why it probably happened:** 1-3 short bullets, clearly worded as likely reasons.\n"
    "**What to do this week:** a numbered list of 2-4 concrete steps (what to do, where in the "
    "store, and the expected gain or cost in Rs if the data provides it).\n"
    "**How to check it worked:** one sentence.\n"
    "Keep the whole answer under 250 words. For a simple or casual question, skip the headings "
    "and answer in 2-4 plain sentences. If the data does not contain the answer, say so plainly "
    "and say what data would be needed."
)


def _table(df, cols=None):
    return df[cols].to_string(index=False) if cols else df.to_string(index=False)


def build_facts(df, insight_list, metrics, actions, forecast, mape, staff_plan,
                seg_summary, hist_text, assumptions, extras=""):
    """Everything Gemini is allowed to know. Aggregates only - no raw rows are sent."""
    zone_stats = df.groupby("zone").agg(
        visitors=("track_id", "count"),
        avg_dwell_s=("dwell_seconds", "mean"),
        avg_purchase_prob=("purchase_probability", "mean"),
    ).round(2)

    parts = [
        f"=== LIVE STORE DATA ({len(df)} visitors currently tracked) ===",
        f"Average dwell time: {df['dwell_seconds'].mean():.1f} s",
        f"Average predicted purchase likelihood: {df['purchase_probability'].mean()*100:.1f}%",
        f"Likely buyers (>50%): {int((df['purchase_probability'] > 0.5).sum())}",
        f"Gender split: {df['gender'].value_counts().to_dict()}",
        f"Age groups: {df['age_group'].value_counts().to_dict()}",
        "Per-zone statistics:",
        zone_stats.to_string(),
        "",
        f"=== MODEL (trained on synthetic data) === accuracy {metrics['accuracy']}, "
        f"ROC-AUC {metrics['roc_auc']}",
        "System-generated insights:",
        *[f"- {m}" for _, m in insight_list],
        "",
        "=== ASSUMPTIONS USED ===",
        ", ".join(f"{k}: {v}" for k, v in assumptions.items()),
        "",
        "=== REVENUE IMPACT OF ACTIONS (for the live visitors; actions overlap, not additive) ===",
        _table(actions),
        "",
        "=== 90-DAY HISTORY (synthetic) ===",
        hist_text,
        "",
        f"=== 7-DAY FOOTFALL FORECAST (back-test error {mape:.1f}% MAPE) ===" if mape is not None
        else "=== 7-DAY FOOTFALL FORECAST ===",
        _table(forecast, ["Day", "Forecast", "Low", "High"]),
        "",
        "=== STAFFING PLAN ===",
        _table(staff_plan),
        "",
        "=== CUSTOMER SEGMENTS (live visitors) ===",
        _table(seg_summary),
        "",
        extras,
    ]
    return "\n".join(parts)


def _parse_json_list(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON list found in response")
    return json.loads(text[start:end + 1])


def _generate(client, contents, config):
    """Call the AI with automatic retries; switch to a backup model if the first is busy."""
    busy_codes = ("503", "UNAVAILABLE", "429", "overloaded")
    last_error, was_busy = None, False
    for model in [MODEL_NAME] + FALLBACK_MODELS:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as e:
                last_error = e
                text = str(e)
                if any(code in text for code in busy_codes) or "500" in text:
                    was_busy = True
                    time.sleep(2 * (attempt + 1))  # wait 2s, 4s, 6s then retry
                else:
                    break  # e.g. model not found (404): skip straight to the next model
    if was_busy:
        raise RuntimeError("The AI service is very busy right now. Please wait a minute "
                           "and try again.")
    raise last_error


def generate_plan(api_key, facts):
    """Returns a list of action dicts, ordered by priority."""
    client = genai.Client(api_key=api_key)
    response = _generate(
        client,
        contents=f"DATA:\n{facts}\n\nProduce the prioritized action plan now.",
        config=types.GenerateContentConfig(
            system_instruction=PLAN_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    plan = _parse_json_list(response.text)
    return sorted(plan, key=lambda x: x.get("priority", 99))


def chat(api_key, facts, history, question):
    """history: list of {"role": "user"|"model", "text": str}. Returns the reply text."""
    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
        for m in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
    response = _generate(
        client,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=CHAT_PROMPT + "\n\nDATA:\n" + facts,
            temperature=0.4,
        ),
    )
    return response.text