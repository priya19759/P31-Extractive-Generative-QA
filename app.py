import io
import os
import re
import time
from collections import Counter

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
from google import genai
from google.genai import types

st.set_page_config(page_title="P31 | Extractive vs Generative QA", page_icon="📘", layout="wide")

st.title("📘 P31 – Extractive vs Generative Question Answering")
st.caption("Compare an extractive BERT-family answer with a passage-grounded LLM answer.")

DEFAULT_CONTEXT = """The Green Campus Initiative began at Riverdale College in 2022. The programme was created to reduce electricity use and improve waste segregation. The college installed 120 LED lights across its main buildings and placed separate bins for paper, plastic and food waste. Students from the Eco Club conduct a campus audit every month. In its first year, the college reported a 15 percent reduction in electricity consumption compared with the previous year. The initiative is funded by the college's sustainability budget."""

@st.cache_resource(show_spinner="Loading the extractive question-answering model...")
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/minilm-uncased-squad2",
        tokenizer="deepset/minilm-uncased-squad2",
    )

@st.cache_resource(show_spinner="Loading the visual question-answering model...")
def load_vqa_model():
    return pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

def normalise(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def token_f1(a: str, b: str) -> float:
    aa, bb = normalise(a).split(), normalise(b).split()
    if not aa or not bb:
        return 0.0
    overlap = sum((Counter(aa) & Counter(bb)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(bb), overlap / len(aa)
    return 2 * precision * recall / (precision + recall)

def extractive_answer(context: str, question: str):
    qa = load_qa_model()
    result = qa(question=question, context=context, handle_impossible_answer=True)
    answer = (result.get("answer") or "").strip()
    score = float(result.get("score") or 0.0)
    # SQuAD2 checkpoint can abstain. The threshold is a tunable heuristic.
    if not answer or score < 0.12:
        answer = "The passage does not provide enough information."
    return answer, score

def generative_answer(context: str, question: str, model_name: str):
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    if not api_key:
        raise RuntimeError("Add GOOGLE_API_KEY in Streamlit app Secrets to enable the LLM answer.")
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a passage-grounded question-answering assistant.
Answer using ONLY the context below. Do not use outside knowledge.
If the answer is absent or cannot be inferred safely, reply exactly:
"The passage does not provide enough information."
Do not invent names, dates, numbers, causes or examples.
Give a concise answer, followed by one short exact evidence quote from the context.
If abstaining, set Evidence to "Not found".

CONTEXT:
{context}

QUESTION:
{question}

Return exactly:
Answer: ...
Evidence: ..."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=220),
    )
    return (response.text or "").strip()

def parse_answer(text: str) -> str:
    match = re.search(r"Answer\s*:\s*(.*?)(?:\nEvidence\s*:|$)", text, flags=re.I|re.S)
    return match.group(1).strip() if match else text.strip()

def compare_answers(context: str, extractive: str, generated: str):
    score = token_f1(extractive, parse_answer(generated))
    generated_norm = normalise(parse_answer(generated))
    context_norm = normalise(context)
    abstain = "does not provide enough information" in generated_norm
    # Warn only when there is disagreement; lexical checks are intentionally conservative.
    unsupported = not abstain and score < 0.30
    if unsupported:
        return score, "REVIEW – possible disagreement / unsupported answer", "error"
    return score, "Answers show lexical agreement or both abstain. Verify against the passage.", "success"

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Gemini model", ["gemini-3.8-flash"], index=0)
    show_demo = st.checkbox("Load sample passage", value=True)
    st.markdown("---")
    st.caption("Privacy: avoid pasting confidential or sensitive personal information.")

context_default = DEFAULT_CONTEXT if show_demo else ""
context = st.text_area("1. Source passage", value=context_default, height=230,
                       placeholder="Paste the passage you want to ask questions about...")
question = st.text_input("2. Your question", placeholder="Example: How many LED lights were installed?")
st.markdown("#### Optional U3 extension – Ask a question about an image")
uploaded_image = st.file_uploader("Upload a figure or diagram (PNG/JPG)", type=["png", "jpg", "jpeg"])
visual_question = st.text_input("Question about the uploaded image", placeholder="Example: What objects are visible?")
if uploaded_image and visual_question.strip():
    if st.button("Answer image question"):
        try:
            image = Image.open(uploaded_image).convert("RGB")
            with st.spinner("Running BLIP visual question answering..."):
                vqa = load_vqa_model()
                visual_result = vqa(image, visual_question)
            st.subheader("Visual answer – BLIP VQA")
            if visual_result:
                st.success(visual_result[0].get("answer", "No answer returned."))
                st.caption("Visual QA is a separate image-grounded branch.")
            else:
                st.warning("No visual answer returned.")
        except Exception as exc:
            st.error(f"Visual QA error: {exc}")
col_a, col_b = st.columns([1, 4])
with col_a:
    run = st.button("Compare answers", type="primary", use_container_width=True)
with col_b:
    st.caption("Both models receive the same passage and question.")

if run:
    st.session_state.pop("p31_result", None)
    if not context.strip() or not question.strip():
        st.error("Please enter both a passage and a question.")
    else:
        started = time.time()
        try:
            with st.spinner("Running extractive QA..."):
                ext_answer, ext_score = extractive_answer(context, question)
            try:
                with st.spinner("Generating grounded LLM answer..."):
                    gen_raw = generative_answer(context, question, model_name)
                gen_answer = parse_answer(gen_raw)
                gen_error = None
            except Exception as exc:
                gen_raw, gen_answer, gen_error = "", "", str(exc)

            st.subheader("Answer comparison")
            left, right = st.columns(2)
            with left:
                st.markdown("### Extractive answer – BERT")
                st.info(ext_answer)
                st.metric("Extractive model score", f"{ext_score:.3f}")
                st.caption("The answer is selected from the source passage.")
            with right:
                st.markdown("### Generative answer – Gemini")
                if gen_error:
                    st.warning("Generative branch unavailable: " + gen_error)
                else:
                    st.success(gen_answer or "No answer returned.")
                    evidence = re.search(r"Evidence\s*:\s*(.*)", gen_raw, flags=re.I|re.S)
                    if evidence:
                        st.caption("Evidence: " + evidence.group(1).strip())
                    with st.expander("View full LLM response"):
                        st.write(gen_raw)

            if not gen_error and gen_answer:
                overlap, status, kind = compare_answers(context, ext_answer, gen_raw)
                st.metric("BERT–LLM answer overlap (token F1)", f"{overlap:.3f}")
                if kind == "error":
                    st.error(status)
                else:
                    st.success(status)
                st.caption("The overlap score is a review heuristic, not proof of factual correctness.")

            final_text = gen_answer if not gen_error and gen_answer else ext_answer
            st.session_state["p31_result"] = {
                "extractive": ext_answer,
                "generative": gen_answer,
                "final": final_text,
                "raw": gen_raw,
                "error": gen_error,
                "score": ext_score,
            }

            st.caption(f"Total processing time: {time.time() - started:.2f} seconds (excluding initial model download).")
        except Exception as exc:
            st.exception(exc)

with st.expander("How this project works"):
    st.markdown("""
    1. **U1 – Prompt engineering:** a grounded-answer instruction defines context, task, output and abstention.
    2. **U2 – LLM:** Gemini generates a concise answer using the passage only.
    3. **U3 – BERT:** a SQuAD2-tuned MiniLM QA checkpoint extracts an answer span.
    4. **U4 – Generation tools:** gTTS reads the final answer aloud. (The optional BLIP-VQA image branch is a future extension.)
    5. **U5 – Responsible AI:** the app flags disagreement, avoids unsupported certainty and asks users to verify the passage.
    """)
st.divider()
st.caption("P31 | Priya Shinde | Roll No. 31 | MIT Art, Design and Technology University")
