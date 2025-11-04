import os
import random
import json
import io # NEW: Import io for BytesIO
import librosa
import numpy as np
import onnxruntime
import soundfile as sf
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
# Removed imports: unicodedata (only needed for Cangjie/Chinese)

# Global variables: only kakasi and dicta removed because they are no longer needed
_kakasi = None # Kept as placeholder
_dicta = None # Kept as placeholder

# --- Constants ---
S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
# Keep only Dutch and the minimum necessary keys (like 'en' for default voice download)
SUPPORTED_LANGUAGES = {
  "nl": "Dutch",
}


# --- Utility Class for Repetition Penalty (Kept) ---

class RepetitionPenaltyLogitsProcessor:
    """
    Applies a repetition penalty to the logits.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Process logits based on input IDs and the penalty factor.
        """
        # Ensure input_ids is 2D (batch_size, sequence_length) for consistency
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]

        # Get the scores of the tokens that have already been generated
        score = np.take_along_axis(scores, input_ids, axis=1)

        # Apply penalty: if score < 0, multiply; otherwise, divide
        score = np.where(score < 0, score * self.penalty, score / self.penalty)

        # Update the scores with the penalized values
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


# --- Multilingual Text Processing Functions (Highly simplified) ---

def prepare_language(txt: str, language_id: str):
    """
    Performs language-specific text processing and prepends the language token to the text.
    """
    
    # Prepend language token
    if language_id and language_id.lower() in SUPPORTED_LANGUAGES:
        # FIX: Added a space after the language token for correct tokenization.
        txt = f"[{language_id.lower()}] {txt}" 
    return txt


# --- Main Synthesizer Class ---

class ChatterboxOnnx:
    """
    A standalone class for performing text-to-speech synthesis using the
    Chatterbox ONNX models, optimized for Dutch.
    """

    def __init__(self, quantized: bool = True,
                 cache_dir: str = os.path.expanduser("~/.cache/chatterbox_onnx")):
        """
        Initialize the ChatterboxOnnx synthesizer and prepare tokenizer, model files, and ONNX inference sessions.
        """
        self.quantized = quantized
        self.model_id = "onnx-community/chatterbox-multilingual-ONNX" 
        self.output_dir = cache_dir

        self.repetition_penalty = 1.2
        self.repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=self.repetition_penalty)

        print(f"Initializing ChatterboxSynthesizer. Model files will be cached in '{cache_dir}'...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, 'onnx'), exist_ok=True)

        self.tokenizer = self._load_tokenizer()

        # Loading the models
        self.speech_encoder_session, \
            self.embed_tokens_session, \
            self.llama_with_past_session, \
            self.cond_decoder_session = self._load_models()

        # These parameters must match the LLM architecture
        self.num_hidden_layers = 30
        self.num_key_value_heads = 16
        self.head_dim = 64

    def _load_tokenizer(self) -> Tokenizer:
        """
        Load the model's tokenizer.json from the Hugging Face Hub and return a Tokenizer instance.
        """
        try:
            # 1. Download the tokenizer.json file
            tokenizer_path = hf_hub_download(
                repo_id=self.model_id,
                filename="tokenizer.json",
                local_dir=self.output_dir
            )

            # 2. Load the tokenizer using the dedicated tokenizers library
            return Tokenizer.from_file(tokenizer_path)

        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

    def _download_and_get_session(self, filename: str) -> onnxruntime.InferenceSession:
        """Downloads an ONNX file and creates an InferenceSession."""
        path = hf_hub_download(
            repo_id=self.model_id,
            filename=filename,
            local_dir=self.output_dir,
            subfolder='onnx'
        )

        # The data file for the ONNX model is also downloaded here
        hf_hub_download(
            repo_id=self.model_id,
            filename=filename.replace(".onnx", ".onnx_data"),
            local_dir=self.output_dir,
            subfolder='onnx'
        )

        return onnxruntime.InferenceSession(path)

    def _load_models(self):
        """
        Download the Chatterbox ONNX model files and create InferenceSession objects.
        """
        model_files = [
            "speech_encoder.onnx",  # -> speech_encoder_session
            "embed_tokens.onnx",  # -> embed_tokens_session
            "language_model_q4.onnx" if self.quantized else "language_model.onnx",
            "conditional_decoder.onnx"  # -> cond_decoder_session
        ]
        
        sessions = []
        for file in model_files:
            print(f"Loading {file}...")
            sessions.append(self._download_and_get_session(file))

        return sessions

    def _generate_waveform(self, text: str,
                           cond_emb, prompt_token, ref_x_vector, prompt_feat,
                           max_new_tokens: int,
                           exaggeration: float,
                           speech_tokens=None,
                           language_id: str = None):

        """
         Generate a waveform conditioned on text and speaker embeddings, optionally generating missing speech tokens.
        """
        if speech_tokens is None:
            
            # Apply multilingual text preparation (now only language-token prepend)
            if language_id:
                text = prepare_language(text, language_id)
            
            # 1. Tokenize Text Input
            encoding = self.tokenizer.encode(text)
            input_ids = np.array([encoding.ids], dtype=np.int64)

            # Calculate position IDs for the text tokens
            position_ids = np.where(
                input_ids >= START_SPEECH_TOKEN,
                0,
                np.arange(input_ids.shape[1])[np.newaxis, :] - 1
            )

            ort_embed_tokens_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "exaggeration": np.array([exaggeration], dtype=np.float32)
            }

            generate_tokens = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

            # --- Generation Loop using kv_cache ---
            for i in range(max_new_tokens):

                # --- Embed Tokens ---
                inputs_embeds = self.embed_tokens_session.run(None, ort_embed_tokens_inputs)[0]

                if i == 0:
                    # Concatenate conditional embedding with text embeddings
                    inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)

                    # Prepare LLM inputs (Attention Mask and Past Key Values)
                    batch_size, seq_len, _ = inputs_embeds.shape

                    # Initialize Past Key Values (Empty cache)
                    past_key_values = {
                        f"past_key_values.{layer}.{kv}": np.zeros(
                            [batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=np.float32)
                        for layer in range(self.num_hidden_layers)
                        for kv in ("key", "value")
                    }
                    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

                # --- Run Language Model (LLama) ---
                llama_with_past_session = self.llama_with_past_session
                logits, *present_key_values = llama_with_past_session.run(None, dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **past_key_values,
                ))

                # Process Logits
                logits = logits[:, -1, :]  # Get logits for the last token
                next_token_logits = self.repetition_processor(generate_tokens[:, -1:], logits)

                # Sample next token (Greedy search: argmax)
                next_token = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)
                generate_tokens = np.concatenate((generate_tokens, next_token), axis=-1)

                # Check for stop token
                if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
                    break

                # Update inputs for next iteration
                position_ids = np.full((input_ids.shape[0], 1), i + 1, dtype=np.int64)
                ort_embed_tokens_inputs["input_ids"] = next_token
                ort_embed_tokens_inputs["position_ids"] = position_ids

                # Update Attention Mask and KV Cache
                attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

            print("Token generation complete.")

            # 2. Concatenate Speech Tokens and Run Conditional Decoder
            # Remove START and STOP tokens
            speech_tokens = generate_tokens[:, 1:-1]
            # Prepend prompt token
            speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)

        cond_incoder_input = {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        }

        # Run the correct decoder session
        wav = self.cond_decoder_session.run(None, cond_incoder_input)[0]
        wav = np.squeeze(wav, axis=0)

        return wav

    def embed_speaker(self, source_audio_path: str):
        # --- Extract speaker embedding from audio ---
        """
        Extract speaker conditioning embeddings, prompt token, speaker embedding vector, and speaker features from a local audio file.
        """
        src_audio, _ = librosa.load(source_audio_path, sr=S3GEN_SR, res_type="soxr_hq")
        src_audio = src_audio[np.newaxis, :].astype(np.float32)
        tgt_cond = {"audio_values": src_audio}
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.speech_encoder_session.run(None, tgt_cond)
        return cond_emb, prompt_token, ref_x_vector, prompt_feat

    def _watermark_and_save(self, wav, output_file_name: str, apply_watermark=False):
        """
        Optionally apply an implicit watermark to an audio waveform and save it to disk.
        """
        if apply_watermark:
            print("Applying audio watermark...")
            try:
                import perth
                watermarker = perth.PerthImplicitWatermarker()
                wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)
            except ImportError:
                print("Warning: 'resemble-perth' not installed. Watermark skipped.")
            except Exception as e:
                print(f"Watermarking failed: {e}")
        # 4. Save Audio File
        sf.write(output_file_name, wav, S3GEN_SR)
        print(f"\nSuccessfully saved generated audio to: {output_file_name}")
        return output_file_name

    # --- NEW HELPER FUNCTION FOR STREAMING ---
    def _save_to_bytesio(self, wav, apply_watermark=False, output_file_name: str = None):
        """
        Optionally apply an implicit watermark to an audio waveform and save it to a BytesIO buffer.
        If output_file_name is provided, saves a copy of the audio to disk.
        """
        if apply_watermark:
            # Watermark logic here
            pass

        # 1. Save Audio to Disk if a name is provided
        if output_file_name:
            sf.write(output_file_name, wav, S3GEN_SR)
            print(f"\nSuccessfully saved generated audio to: {output_file_name}")

        # 2. Save Audio to BytesIO Buffer
        buffer = io.BytesIO()
        # soundfile.write can write to a file-like object
        sf.write(buffer, wav, S3GEN_SR, format='WAV') 
        buffer.seek(0) # Ensure the cursor is at the start of the buffer for reading
        return buffer
    # ----------------------------------------

    def voice_convert(
            self,
            source_audio_path: str,
            target_voice_path: str,
            output_file_name: str = "converted_voice.wav",
            exaggeration: float = 0.5,
            max_new_tokens: int = 512,
            apply_watermark=False
    ):
        """
        Convert a source audio file to sound like a target (reference) voice and save the converted audio.
        """
        print("\n--- Starting ONNX Voice Conversion ---")
        print(f"Source: {source_audio_path}\nTarget: {target_voice_path}\nOutput: {output_file_name}")

        # --- Extract speaker embedding from target audio ---
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(target_voice_path)

        # --- Tokenize the source speech ---
        _, src_tokens, _, _ = self.embed_speaker(source_audio_path)

        # Prepend target prompt token to source tokens for conditioning
        speech_tokens = np.concatenate([prompt_token, src_tokens], axis=1)

        wav = self._generate_waveform(text="",
                                      cond_emb=cond_emb,
                                      prompt_token=src_tokens,
                                      ref_x_vector=ref_x_vector,
                                      prompt_feat=prompt_feat,
                                      max_new_tokens=max_new_tokens,
                                      exaggeration=exaggeration,
                                      speech_tokens=speech_tokens)
        return self._watermark_and_save(wav, output_file_name, apply_watermark)

    def batch_voice_convert(
            self,
            original_audios_folder: str,
            voices_folder: str,
            output_dir: str = "batch_vc_output",
            n_random: int = 2,
    ):
        """
        Perform batch voice cloning by converting selected source WAV files to each reference voice.
        """
        print(f"\n--- Starting Batch Voice Conversion ---")
        os.makedirs(output_dir, exist_ok=True)

        # Gather reference and source voices
        src_files = [os.path.join(original_audios_folder, f) for f in os.listdir(original_audios_folder) if
                     f.lower().endswith('.wav')]
        ref_files = [os.path.join(voices_folder, f) for f in os.listdir(voices_folder) if
                     f.lower().endswith('.wav')]

        if not ref_files or not src_files:
            print("No valid .wav files found in input folders.")
            return

        for ref_path in ref_files:
            ref_name = os.path.splitext(os.path.basename(ref_path))[0]
            selected_src = random.sample(src_files, min(n_random, len(src_files)))

            for src_path in selected_src:
                src_name = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{ref_name}_clone_{src_name}.wav"
                out_path = os.path.join(output_dir, out_name)

                try:
                    self.voice_convert(
                        source_audio_path=src_path,
                        target_voice_path=ref_path,
                        output_file_name=out_path,
                    )
                except Exception as e:
                    print(f"Error processing {src_name} -> {ref_name}: {e}")

    def synthesize(
            self,
            text: str,
            language_id: str = "nl", # Default set to Dutch
            target_voice_path: str = None, 
            max_new_tokens: int = 512,
            exaggeration: float = 0.5,
            output_file_name: str = "output.wav",
            apply_watermark: bool = False,
    ):
        """
        Synthesize speech from text using a target voice and save the resulting WAV file.
        """
        print("\n--- Starting Text-to-Audio Inference ---")

        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            # Only check if it is Dutch, otherwise a warning
            if language_id.lower() != 'nl':
                print(f"Warning: Unsupported language_id '{language_id}'. Only 'nl' is supported in this stripped script. Skipping preprocessing.")
                language_id = None
            else:
                 # Ensure 'nl' is used, regardless of case
                language_id = 'nl'

        if not target_voice_path:
            # Download the default voice
            target_voice_path = hf_hub_download(
                repo_id=self.model_id,
                filename="default_voice.wav",
                local_dir=self.output_dir
            )
            print(f"Using default voice: {target_voice_path}")

        # 2. Generate Waveform
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(target_voice_path)
        wav = self._generate_waveform(text,
                                      cond_emb, prompt_token, ref_x_vector, prompt_feat,
                                      max_new_tokens, exaggeration,
                                      language_id=language_id) 
        self._watermark_and_save(wav, output_file_name, apply_watermark)

    # --- NEW PUBLIC METHOD FOR STREAMING ---
    def synthesize_to_bytesio(
            self,
            text: str,
            language_id: str = "nl", 
            target_voice_path: str = None, 
            max_new_tokens: int = 512,
            exaggeration: float = 0.5,
            apply_watermark: bool = False,
            output_file_name: str = None, # Added argument for saving
    ) -> io.BytesIO:
        """
        Synthesizes speech and returns the audio as an io.BytesIO buffer.
        """
        print(f"\n--- Starting Text-to-Audio Inference to BytesIO ---")

        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            if language_id.lower() != 'nl':
                language_id = None
            else:
                language_id = 'nl'

        if not target_voice_path:
            target_voice_path = hf_hub_download(
                repo_id=self.model_id,
                filename="default_voice.wav",
                local_dir=self.output_dir
            )

        # 2. Generate Waveform
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(target_voice_path)
        wav = self._generate_waveform(text,
                                      cond_emb, prompt_token, ref_x_vector, prompt_feat,
                                      max_new_tokens, exaggeration,
                                      language_id=language_id) 
        
        # Write to BytesIO buffer and save to file if output_file_name is set
        return self._save_to_bytesio(wav, apply_watermark, output_file_name)
    # ----------------------------------------
    
    def batch_synthesize(
            self,
            text: str,
            voice_folder_path: str,
            language_id: str = "nl", # Default set to Dutch
            exaggeration_range: tuple[float, float, float] = (0.5, 0.7, 0.1),  # (start, stop, step)
            max_new_tokens: int = 512,
            output_dir: str = "batch_output",
            apply_watermark: bool = False,
    ):
        """
        Perform batch text-to-speech synthesis using multiple reference voices and a range of exaggeration values.
        """
        print(f"\n--- Starting Batch Synthesis for text: '{text[:40]}...' ---")
        
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            if language_id.lower() != 'nl':
                print(f"Warning: Unsupported language_id '{language_id}'. Only 'nl' is supported in this stripped script. Skipping preprocessing.")
                language_id = None
            else:
                 # Ensure 'nl' is used, regardless of case
                language_id = 'nl'

        os.makedirs(output_dir, exist_ok=True)

        # 1. Prepare exaggeration values
        start, stop, step = exaggeration_range
        if step > 0 and start <= stop:
            exaggeration_values = np.arange(
                start,
                stop + step / 2,  # Add half step for float precision
                step
            ).round(2).tolist()
        else:
            exaggeration_values = [start]

        print(f"Testing exaggeration values: {exaggeration_values}")

        # 2. Find all WAV files
        try:
            voice_files = [
                os.path.join(voice_folder_path, f)
                for f in os.listdir(voice_folder_path)
                if f.lower().endswith('.wav')
            ]
        except FileNotFoundError:
            print(f"Error: Voice folder path not found: '{voice_folder_path}'. Aborting batch.")
            return

        if not voice_files:
            print(f"Error: No .wav files found in '{voice_folder_path}'. Aborting batch.")
            return

        total_generations = len(voice_files) * len(exaggeration_values)
        print(f"Found {len(voice_files)} reference voices. Will perform {total_generations} total generations.")

        # 3. Main Batch Loop
        for voice_path in voice_files:
            voice_name = os.path.splitext(os.path.basename(voice_path))[0]

            print(f"\nProcessing voice: {voice_name}")

            cond_emb, prompt_token, ref_x_vector, prompt_feat = self.embed_speaker(voice_path)

            for ex_val in exaggeration_values:
                print(f"  > Generating with exaggeration={ex_val:.2f}...")

                output_name = f"{voice_name}_exag{ex_val:.2f}.wav"
                output_file_path = os.path.join(output_dir, output_name)

                try:
                    # Generate Waveform
                    wav = self._generate_waveform(text,
                                                  cond_emb, prompt_token, ref_x_vector, prompt_feat,
                                                  max_new_tokens, ex_val,
                                                  language_id=language_id) 
                    self._watermark_and_save(wav, output_file_path, apply_watermark)

                except Exception as e:
                    print(f"  Error generating {output_name}: {e}")
                    continue

        print("\n--- Batch Synthesis Complete ---")

    def debug_info(self):
        """Print detailed ONNX session information and sample IO shapes."""

        def print_session_info(session, name):
            print(f"\n===== {name} =====")
            print("Providers:", session.get_providers())
            print("Inputs:")
            for inp in session.get_inputs():
                print(f" name={inp.name}, shape={inp.shape}, type={inp.type}")
            print("Outputs:")
            for out in session.get_outputs():
                print(f" name={out.name}, shape={out.shape}, type={out.type}")

        print("\n==================== DEBUG INFO ====================")
        print(f"Model ID: {self.model_id}")
        print(f"Cache directory: {self.output_dir}")

        sessions = [
            (self.speech_encoder_session, "Speech Encoder"),
            (self.embed_tokens_session, "Embed Tokens"),
            (self.llama_with_past_session, "Language Model"),
            (self.cond_decoder_session, "Conditional Decoder"),
        ]

        for sess, name in sessions:
            try:
                print_session_info(sess, name)
            except Exception as e:
                print(f"Error inspecting {name}: {e}")

        # Optional: Run a fake forward pass to inspect output shapes
        try:
            import librosa
            # Use self.model_id to get the correct default voice
            wav, _ = librosa.load(
                hf_hub_download(repo_id=self.model_id, filename="default_voice.wav", local_dir=self.output_dir),
                sr=24000)
            wav = wav[np.newaxis, :].astype(np.float32)
            print("\nRunning speech encoder on default voice for shape check...")
            outs = self.speech_encoder_session.run(None, {"audio_values": wav})
            for i, out in enumerate(outs):
                print(f" Output[{i}] shape={np.array(out).shape}, dtype={np.array(out).dtype}")
        except Exception as e:
            print(f"Could not run shape check: {e}")

        print("====================================================\n")


if __name__ == "__main__":
    print("This file has been adapted to work with app.py for streaming.")