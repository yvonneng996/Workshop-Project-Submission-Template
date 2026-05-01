# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:29:06 2026

@author: yvonneng
"""
import os
import joblib
import anthropic
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from pathlib import Path
# ── API Setup ─────────────────────────────────────────────────────────
os.environ["ANTHROPIC_API_KEY"] = "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"
API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-6"

# ── Initialize Anthropic Client ───────────────────────────────────────
client = anthropic.Anthropic(api_key=API_KEY)
BASE_DIR = Path(__file__).parent  
def classify_with_claude(vessel_density, image_list=None):
    """
    Load classification model, predict Efron severity grade based on 
    vessel density, then send result to Claude via Anthropic API for 
    clinical explanation and recommendation.
    """
    model = joblib.load(BASE_DIR / "Best_Model" / "ordinal_logistic_model.pkl")
    # ── Step 1: Load classification model ────────────────────────────
        #"C:\\Users\\yvonneng\\Downloads\\project_mtech\\Best_Model\\ordinal_logistic_model.pkl")
    
    # ── Step 2: Predict severity grade ───────────────────────────────
    X = np.array([[vessel_density]])
    predicted_grade = model.predict(X)[0]
    
    # Map grade to Efron label
    grade_labels = {
        0: "Normal",
        1: "Trace",
        2: "Mild",
        3: "Moderate",
        4: "Severe"
    }
    grade_label = grade_labels.get(int(predicted_grade), "Unknown")

    # ── Step 3: Build image payload ───────────────────────────────────
    images_payload = []
    if image_list is not None:
        if not isinstance(image_list, list):
            image_list = [image_list]
        for img in image_list:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            images_payload.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64
                }
            })

    # ── Step 4: System prompt ─────────────────────────────────────────
    system_prompt = """You are an experienced ophthalmology clinician 
specialising in conjunctival hyperaemia assessment and dry eye disease 
management. Your role is to provide clear, concise and clinically 
accurate interpretations of conjunctival hyperaemia severity grades 
based on the Efron grading scale to assist clinicians in making 
informed treatment decisions.

The Efron grading scale for conjunctival hyperaemia is defined as follows:

- Grade 0 (Normal): No visible vessel engorgement. The conjunctiva 
  appears white and clear with fine vessels barely visible.

- Grade 1 (Trace): Barely perceptible vessel engorgement. A very 
  slight pinkish appearance with fine vessels slightly more visible 
  than normal but not prominent.

- Grade 2 (Mild): Some vessel engorgement visible. A noticeable 
  pinkish-red appearance with clearly visible fine and some larger 
  vessels distributed across the conjunctiva.

- Grade 3 (Moderate): Obvious vessel engorgement across the 
  conjunctiva. A clearly red appearance with prominent dilated 
  vessels, both fine and thick, distributed diffusely across the 
  conjunctival surface.

- Grade 4 (Severe): Severe vessel engorgement with a deep red 
  appearance. Markedly dilated and congested vessels covering the 
  entire conjunctival surface, associated with significant ocular 
  surface inflammation.

Important rules:
- Do NOT re-grade or override the provided severity grade
- Base your explanation on the provided grade and vessel density value
- Keep your response clinically relevant, factual and concise
- Use language suitable for a clinical report"""

    # ── Step 5: User prompt ───────────────────────────────────────────
    user_prompt = f"""The automated classification model has analysed 
the patient's conjunctival image and produced the following results:

- Predicted vessel density: {vessel_density:.4f}
- Predicted Efron severity grade: Grade {predicted_grade} ({grade_label})

Please provide the following based on these results:

1. Clinical Interpretation
   Explain what Grade {predicted_grade} ({grade_label}) means clinically 
   and how the vessel density of {vessel_density:.4f} supports this grade.
   Comment on whether the vessel density value is consistent with 
   the expected range for this severity level.

2. Vascular Features
   Describe the key vascular features expected at this severity level 
   including thin vessel density, thick vessel density, vessel 
   distribution pattern and area coverage across the conjunctival surface.
   Comment on whether the provided conjunctival image visually supports 
   the predicted grade based on these features.

3. Severity Comparison
   Briefly explain how this grade differs from the grade immediately 
   below and above it so the clinician understands where the patient 
   sits within the overall severity spectrum.

4. Management Recommendation
   Suggest appropriate clinical management or treatment considerations 
   for a patient presenting at Grade {predicted_grade} ({grade_label}).
   Include both pharmacological and non-pharmacological options 
   where relevant.

5. Lifestyle and Patient Advisory
   Provide brief lifestyle or self-care advice the clinician can 
   pass on to the patient to help manage or reduce conjunctival 
   hyperaemia at this severity level.

6. Follow-up Actions
   Outline recommended follow-up actions the clinician should 
   consider based on this severity grade.
   Specify a suggested follow-up timeframe and any monitoring 
   parameters that should be tracked at the next visit.

7. Red Flags
   List any warning signs or symptoms that would indicate the 
   condition is worsening and should prompt an earlier review 
   or urgent referral to a specialist.

Use clear and simple language suitable for a clinical report.
Each section should be 2 to 3 sentences minimum.
Reply in plain text."""

    # ── Step 6: Build message content ────────────────────────────────
    content = []
    
    # Add images first if any
    content.extend(images_payload)
    
    # Add text prompt
    content.append({"type": "text", "text": user_prompt})

    # ── Step 7: Call Claude via Anthropic API ─────────────────────────
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    # ── Step 8: Parse and return response ────────────────────────────
    llm_output = response.content[0].text

    return {
        "vessel_density": vessel_density,
        "predicted_grade": int(predicted_grade),
        "grade_label": grade_label,
        "llm_explanation": llm_output
    }

