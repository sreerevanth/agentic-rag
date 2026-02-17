import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = "embeddings"

st.set_page_config(
    page_title="CYNOTE",
    layout="centered"
)

# =============================
# THEME TOGGLE
# =============================
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)

# =============================
# CYBER CYAN THEME
# =============================
if dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #020617;
        color: #22d3ee;
    }

    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #22d3ee33;
    }

    section[data-testid="stSidebar"] * {
        color: #22d3ee !important;
        font-weight: 600;
    }

    h1, h2, h3, h4 {
        color: #22d3ee;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    p, li, span {
        color: #22d3ee !important;
        font-weight: 600;
    }

    input {
        background-color: #020617 !important;
        color: #22d3ee !important;
        border: 1px solid #22d3ee;
        border-radius: 12px;
    }

    button {
        background-color: #020617 !important;
        color: #22d3ee !important;
        border: 2px solid #22d3ee;
        border-radius: 999px;
        font-weight: 800;
        box-shadow: 0 0 12px #22d3ee66;
    }

    button:hover {
        background-color: #22d3ee !important;
        color: #020617 !important;
        box-shadow: 0 0 20px #22d3ee;
    }

    div[data-baseweb="radio"] label {
        background-color: #020617;
        border-radius: 999px;
        padding: 10px 18px;
        margin: 6px;
        border: 1px solid #22d3ee;
        color: #22d3ee !important;
        font-weight: 700;
        box-shadow: 0 0 10px #22d3ee44;
    }

    div[data-baseweb="radio"] label:hover {
        background-color: #22d3ee;
        color: #020617 !important;
        box-shadow: 0 0 18px #22d3ee;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #f9fafb;
        color: #111827;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    section[data-testid="stSidebar"] * {
        color: #111827;
        font-weight: 600;
    }
    h1, h2, h3, h4 {
        color: #111827;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🧠 CYNOTE | System Info")
st.sidebar.markdown("""
• Works offline  
• Uses only your notes  
• No hallucination  
• Exam-oriented output  
""")

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Answer Mode",
    [
        "Explain (2 Marks)",
        "Explain (5 Marks)",
        "Explain (10 Marks)",
        "Show All Related Content",
        "Build Topic Notes (Exam)"
    ]
)

# =============================
# MAIN UI
# =============================
st.markdown(
    "<h1 style='text-align:center;'>⚡ CYNOTE</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Offline • Exam-Oriented • Zero Hallucination</p>",
    unsafe_allow_html=True
)

st.markdown("---")

question = st.text_input("💡 Enter your question")

# =============================
# ASK BUTTON
# =============================
if st.button("🚀 Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=6)

    # =============================
    # 10 MARKS / TOPIC NOTES
    # =============================
    if mode in ["Explain (10 Marks)", "Build Topic Notes (Exam)"]:
        st.subheader("📘 10-Mark Answer (Topic-wise from Notes)")

        topics = [
            "software specification",
            "software development",
            "software validation",
            "software evolution"
        ]

        for topic in topics:
            topic_docs = db.similarity_search(topic, k=5)
            seen = set()

            for d in topic_docs:
                text = d.page_content.strip()
                if len(text) < 50:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)

                meta = d.metadata
                source = meta.get("source", "Unknown")
                page = meta.get("page_label", meta.get("page", "N/A"))

                st.markdown(f"### 🔹 {topic.title()}")
                st.markdown(f"📄 *{source} | Page {page}*")
                st.write(text)

    # =============================
    # OTHER MODES
    # =============================
    else:
        context = "\n\n".join([d.page_content for d in docs][:4])

        llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

        max_tokens = 120 if "2 Marks" in mode else 250

        answer = llm(
            f"Answer using ONLY the notes below.\n\nNOTES:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:",
            max_new_tokens=max_tokens
        )[0]["generated_text"]

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📌 Sources")
        shown = set()
        for d in docs:
            meta = d.metadata
            source = meta.get("source", "Unknown")
            page = meta.get("page_label", meta.get("page", "N/A"))
            key = f"{source}-{page}"
            if key in shown:
                continue
            shown.add(key)
            st.write(f"- {source}, Page {page}")
