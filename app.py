import argparse

import gradio as gr

from audiodiffusion import AudioDiffusion


def generate_spectrogram_audio_and_loop(model_id, output_path, sample_rate, n_fft, hop_length, top_db, steps):
    audio_diffusion = AudioDiffusion(model_id=model_id, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, top_db=top_db)
    image, (sample_rate, audio) = audio_diffusion.generate_spectrogram_and_audio(steps)
    loop = AudioDiffusion.loop_it(audio, sample_rate)
    if loop is None:
        loop = audio
    return image, (sample_rate, audio), (sample_rate, loop)


demo = gr.Interface(
    fn=generate_spectrogram_audio_and_loop,
    title="Audio Diffusion",
    description="Generate audio using Huggingface diffusers.\
        This takes about 20 minutes without a GPU, so why not make yourself a \
            cup of tea in the meantime? (Or try the teticio/audio-diffusion-ddim-256 \
                model which is faster.)",
    inputs=[
        gr.Dropdown(
            label="Model",
            choices=[
                "teticio/audio-diffusion-256",
                "teticio/audio-diffusion-breaks-256",
                "teticio/audio-diffusion-instrumental-hiphop-256",
                "teticio/audio-diffusion-ddim-256",
            ],
            value="teticio/audio-diffusion-256",
        ),
        gr.Textbox(label="Output path"),
        gr.Number(label="Sample rate", value=22050),
        gr.Number(label="n_fft", value=2048),
        gr.Number(label="hop_length", value=512),
        gr.Number(label="top_db", value=80),
        gr.Number(label="steps", value=None),
    ],
    outputs=[
        gr.Image(label="Mel spectrogram", image_mode="L"),
        gr.Audio(label="Audio"),
        gr.Audio(label="Loop"),
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--server", type=int)
    args = parser.parse_args()
    demo.launch(server_name=args.server or "0.0.0.0", server_port=args.port)
