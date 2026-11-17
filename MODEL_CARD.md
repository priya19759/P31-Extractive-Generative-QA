# Model Card – P31 Extractive vs Generative QA

## Intended use
Educational question answering over user-supplied passages and comparison of extractive and generative outputs.

## Components
- Extractive model: `deepset/minilm-uncased-squad2` (verify its model-card licence and intended use before redistribution/deployment).
- Generative model: Gemini API model selected in the UI.
- Visual QA: `Salesforce/blip-vqa-base` (optional image question branch; verify upstream licence).
- TTS: gTTS.

## Inputs and outputs
Input: passage and question. Output: extracted span, generated answer/evidence, disagreement warning and optional audio.

## Risks
Hallucination, incorrect span selection, paraphrase-related false alarms, domain/language bias, and exposure of sensitive text to an external provider.

## Safeguards
Context-only prompt, abstention instruction, disagreement warning, privacy notice and human verification.

## Limitations
The disagreement score is lexical, not semantic entailment. Model confidence is not calibrated correctness. Visual QA is an optional branch and may be constrained by free-host memory and startup time.

## Data
Use public, original or appropriately permitted passages. Do not publish private user inputs.

## Licence
Check the upstream model and API terms before use. The project code is provided for educational use; third-party components retain their own licences.
