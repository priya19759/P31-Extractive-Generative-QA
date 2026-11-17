# P31 – Extractive vs Generative Question Answering

A Streamlit application that compares:
- **Extractive QA:** `deepset/minilm-uncased-squad2` selects an answer span from the passage.
- **Generative QA:** Gemini produces a concise answer with an evidence quote, using only the passage.
- **Disagreement warning:** token-overlap F1 highlights cases that need manual review.
- **TTS:** gTTS reads the displayed answer aloud.
- **Visual QA:** optional BLIP-VQA image upload and question path.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set `GOOGLE_API_KEY` as an environment variable or in `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "YOUR_KEY"
```

Never commit secrets. Add the key in Streamlit Community Cloud → App → Settings → Secrets.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Sign in to Streamlit Community Cloud and create an app from the repository.
3. Select `app.py` as the entry point.
4. Add `GOOGLE_API_KEY` in the app's Secrets settings.
5. Deploy and test a fresh passage/question.

## Evaluation

`evaluation/benchmark_20.csv` contains a starter 20-item evaluation set. Run both models on all rows, record predictions and calculate EM/F1 with `evaluation/evaluate.py`. Do not report scores until the models have actually been run.

## Important limitations

- The overlap warning is heuristic and cannot prove hallucination.
- Model scores are not calibrated probabilities of correctness.
- The app requires a Gemini API key for the generative branch.
- The BLIP-VQA branch is optional and downloads a separate vision model on first use; test host memory and startup time.
- Do not submit confidential or sensitive personal information.
