import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/ask")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="Pharma RAG", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:wght@500;700&display=swap');
html, body, [class*="css"]  {
  font-family: 'Space Grotesk', sans-serif;
}
.title {
  font-family: 'Fraunces', serif;
  font-size: 40px;
  font-weight: 700;
  letter-spacing: 0.5px;
}
.subtitle {
  font-size: 14px;
  opacity: 0.7;
}
.card {
  background: radial-gradient(120% 120% at 10% 10%, #f7f3ec 0%, #efe6da 55%, #e9dccb 100%);
  border: 1px solid #d9c9b7;
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 10px 24px rgba(44, 36, 29, 0.08);
}
.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #2f261c;
  color: #f5efe7;
  font-size: 12px;
}
.answer {
  font-size: 16px;
  line-height: 1.6;
}
.panel {
  background: #1f1c18;
  color: #f5efe7;
  border-radius: 16px;
  padding: 16px 18px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Pharma RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hybrid router • Pharma suggestions + Web/RSS</div>',
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    query = st.text_area(
        "Ask a question",
        height=130,
        placeholder="e.g., What are recent findings about ocrelizumab in multiple sclerosis?",
    )
    cols = st.columns(4)
    with cols[0]:
        limit = st.slider("Limit", 1, 20, 5)
    with cols[1]:
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.5, 0.05)
    with cols[2]:
        source_type = st.selectbox("Source Type", ["auto", "rss_article", "seed", "pdf"])
    with cols[3]:
        include_source_details = st.checkbox("Show Sources", value=False)

    options = st.columns(2)
    with options[0]:
        include_explanation = st.checkbox("Explain If Empty", value=False)
    with options[1]:
        include_filters = st.checkbox("Show Filters", value=False)

    st.write("")
    submitted = st.button("Ask")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Router**")
    st.markdown("Rules first, LLM if ambiguous.")
    st.markdown("")
    st.markdown("**API**")
    st.markdown(API_URL)
    st.markdown("</div>", unsafe_allow_html=True)

if submitted and query.strip():
    payload = {
        "query": query.strip(),
        "limit": int(limit),
        "score_threshold": float(score_threshold),
    }
    if source_type != "auto":
        payload["source_type"] = source_type
    payload["include_source_details"] = bool(include_source_details)
    payload["include_explanation"] = bool(include_explanation)
    payload["include_filters"] = bool(include_filters)

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    with st.spinner("Thinking..."):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                route = data.get("route_used")
                color = "#2f261c"
                if route == "web":
                    color = "#1f5a7a"
                elif route == "pharma":
                    color = "#2b6b3f"
                elif route == "multi":
                    color = "#6b3f8f"
                st.markdown(
                    f"<span class='pill' style='background:{color}'>route: {route}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='answer'>", unsafe_allow_html=True)
                st.markdown(data.get("answer", ""))
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Request failed: {e}")
