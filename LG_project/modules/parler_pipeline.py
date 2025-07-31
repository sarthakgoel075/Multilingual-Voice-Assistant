# parler_pipeline.py

import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from parler_tts import ParlerTTSForConditionalGeneration

# ---- Configuration ----
parler_dir = "/content/drive/MyDrive/LG_project/parler_tts_model"
mbart_dir = "/content/drive/MyDrive/LG_project/mbart_translation_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load Models ----
model = ParlerTTSForConditionalGeneration.from_pretrained(parler_dir + "/model").to(device)
tokenizer = AutoTokenizer.from_pretrained(parler_dir + "/tokenizer")
description_tokenizer = AutoTokenizer.from_pretrained(parler_dir + "/description_tokenizer")

mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_dir + "/model").to(device)
mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_dir + "/tokenizer")

# ---- Translate Function ----
def translate(text, target_lang_code):
    mbart_tokenizer.src_lang = "en_XX"
    if target_lang_code not in mbart_tokenizer.lang_code_to_id:
        raise ValueError(f"❌ Invalid target language code: {target_lang_code}")
    
    encoded = mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    generated_tokens = mbart_model.generate(
        **encoded,
        forced_bos_token_id=mbart_tokenizer.lang_code_to_id[target_lang_code]
    )
    return mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ---- TTS Function ----
def text_to_speech_parler(text: str, description: str, output_path: str = "output.wav"):
    try:
        desc_enc = {k: v.to(device) for k, v in description_tokenizer(description, return_tensors="pt").items()}
        prompt_enc = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}

        # Debug (optional)
        print(f"🎯 Device Check:")
        for k, v in desc_enc.items():
            print(f"desc_enc[{k}]: {v.device}")
        for k, v in prompt_enc.items():
            print(f"prompt_enc[{k}]: {v.device}")
        print(f"model: {next(model.parameters()).device}")

        with torch.no_grad():
            output = model.generate(
                input_ids=desc_enc["input_ids"],
                prompt_input_ids=prompt_enc["input_ids"]
            )

        audio = output.cpu().numpy().squeeze()
        if audio.size == 0 or np.isnan(audio).any():
            print("❌ No audio generated or array contains NaNs.")
            return

        sf.write(output_path, audio, model.config.sampling_rate)
        print(f"✅ Audio saved to '{output_path}'")

    except Exception as e:
        print(f"❌ Error generating audio: {e}")