# Finetuning

More instructions coming soon.

VibeVoice finetuning works wonders - both for adaptipng VibeVoice to new languages and for better voice cloning of a single voice.

Join the [Discord](https://discord.gg/ZDEYTTRxWG) for support. Also take a look at [voicepowered-ai/VibeVoice-finetuning](https://github.com/voicepowered-ai/VibeVoice-finetuning).

## Notes

* Members of the community have observed that for fine-tuning VibeVoice on a SINGLE voice, it is often beneficial to set `voice_prompt_drop_rate` to `1.0` and avoid use of voice cloning/reference audio all together during inference. This can lead to more natural speech generation. **If you are training on a single speaker I highly recommend you try this, just a note that if you do this voice cloning will not be supported on your finetuned model**

## Example Script

Example script:

```bash
python -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --dataset_name vibevoice/jenny_vibevoice_formatted \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name audio \
    --output_dir finetune_vibevoice_zac \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2.5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing False \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8
```

## Notes

- Multi-speaker (podcast) fine-tuning is supported: see the section below. It has been checked at the token level against inference, not yet validated by the original authors.
- This is an unofficial finetuning implementation, it has not been validated by the original authors.
- The `voice_prompts_column_name` parameter is currently set to `audio` in the example above, which means the same audio file is used for both training data and voice prompts. This is appropriate when you don't have separate voice prompt files. However, if your dataset includes dedicated voice prompt files (short audio clips that capture the target speaker's voice characteristics), you should specify a different column name that contains these separate voice prompt files. For podcast-style training the voice prompt for each speaker is a clip of that speaker from the same episode, outside the training window; `prepare_podcast_jsonl.py` picks one per speaker.
- The dataset text/transcript must be in the format of "Speaker X: text", even if there is only one speaker. Example: `Speaker 1: Hello, how are you?`
- The default dataset is the Jenny (Dioco) dataset. This is a small dataset for testing purposes and each segment is only a few seconds long. The model may struggle to generate long audio with this dataset.

## Multi-speaker (podcast) fine-tuning

A podcast row is one continuous recording of several speakers plus a script with one `Speaker N:` line per turn and **one voice prompt per speaker**:

```json
{"text": "Speaker 0: I heard there is big news?\nSpeaker 1: Yes! ...\nSpeaker 0: Tell me more.",
 "audio": "windows/ep01/000012.wav",
 "voice_prompts": ["prompts/ep01/A_120.40.wav", "prompts/ep01/B_88.10.wav"]}
```

The processor labels prompts `Speaker 0`, `Speaker 1`, ... by position, so the text must use 0-based ids in order of first appearance and `voice_prompts` must be in that order. Pass `--normalize_speaker_ids True` to the trainer and it renumbers whatever ids your rows use (`voice_prompts` may then also be a dict keyed by the original id, e.g. `{"1": "a.wav", "2": "b.wav"}`). Rows with fewer prompts than speakers fail loudly; rows with no prompts are trained prompt-less, like `voice_prompt_drop_rate` does. Cutting a random prompt from the target audio (the single-speaker fallback) is never done for multi-speaker rows.

Everything else in the training loop is unchanged: the whole window is the diffusion target, prompts are input only, and the sequence is exactly the inference prompt followed by the target latents (`tests/test_multispeaker_finetune.py` checks this against the processor).

### Building rows from diarized episodes

```bash
python -m vibevoice.finetune.prepare_podcast_jsonl \
    --episodes episodes.jsonl --out data/podcast \
    --min-seconds 8 --max-seconds 60 --max-gap 2.0 --require-speaker-change \
    --min-confidence 0.8 --dry-run          # counts and hours only; drop --dry-run to cut audio
```

`episodes.jsonl` has one line per episode: `{"id", "audio", "turns": [{"speaker", "start", "end", "text", "confidence"?}], "prompts"?: {speaker: wav}}`. Windows are consecutive turns cut from the original recording, so gaps, overlaps and reactions are real. Long textless turns and low-confidence turns end a window instead of being generated without text. Validation holds out whole episodes.

Then train as in the example above with:

```
--train_jsonl data/podcast/train.jsonl --validation_jsonl data/podcast/validation.jsonl \
--text_column_name text --audio_column_name audio --voice_prompts_column_name voice_prompts \
--normalize_speaker_ids True --voice_prompt_drop_rate 0.2
```

Windows of 60 s are about 450 speech tokens plus prompts; a 24 GB GPU handles the 1.5B model at batch size 1-2 with `--gradient_checkpointing True`. For inference, `demo/inference_from_file.py --normalize_speaker_ids` applies the same renumbering to your script so it matches training.

### New language

The same path teaches a new language: the text tokenizer (Qwen2.5) already covers most scripts byte-wise, so nothing needs to be added to the vocabulary; the LoRA on the language model and the trained diffusion head learn the pronunciation from your audio. Expect to need several hours of clean, accurately transcribed speech, and judge checkpoints by listening.
