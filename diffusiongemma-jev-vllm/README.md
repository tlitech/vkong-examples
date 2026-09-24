# DiffusionGemma structured decisions with Jev and vLLM

Deploy Google's `google/diffusiongemma-26B-A4B-it` as a Jev-compatible structured
decision service. The public service exposes `/v1/systemone`; a pinned vLLM server
runs privately inside the same machine.

DiffusionGemma is a **text diffusion model**, despite the name. This recipe uses
the official 26B total / 4B active MoE checkpoint in BF16 on one H100 80 GB GPU.

## Requirements

- VKong CLI installed and logged in.
- One verified H100 with 80 GB VRAM.
- Around 55 GB of model weights. The first start can take several minutes.
- Review the displayed hourly price before accepting the rental. The config caps
  it at `$4.00/hour`.

The recipe pins:

- model revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`;
- vLLM nightly commit `e9757321527ca1ecd514c07c1418dd2c53da3d19`;
- Linux/amd64 image manifest
  `sha256:e0eee5c5506bea9bfe350f7d99b07dc49e37d42647a128c2a57ff184551fba10`.

## Run through a private local URL

```bash
cd diffusiongemma-jev-vllm
vkong run -C .
```

VKong prints the actual local URL after the model becomes ready. In another
terminal, use that URL:

```bash
VKONG_URL=http://127.0.0.1:<local-port> python3 client.py
```

Health check:

```bash
curl -fsS http://127.0.0.1:<local-port>/health
```

## Deploy with a public URL

The upstream Jev server supports bearer authentication through the `API_KEY`
environment variable. Before exposing this expensive model publicly, attach a
VKong App Secret containing a strong `API_KEY`, then include that Secret bundle
in the App configuration.

```bash
vkong deploy -C .
```

VKong prints the HTTPS App URL. Test it without putting the key in shell history:

```bash
export VKONG_URL=https://<your-app-url>
read -rs API_KEY && export API_KEY
python3 client.py
```

The gateway also supports:

- `POST /v1/chat/completions` for an OpenAI-shaped structured decision;
- `POST /v1/raw/chat/completions` for ordinary DiffusionGemma generation;
- `noul`, `choice`, and `score` question types;
- dependent questions, multiple samples, optional thinking, and image inputs.

## Stop billing

```bash
vkong app stop diffusiongemma-jev-vllm
```

## Sources

- [Google model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [vLLM DiffusionGemma recipe](https://github.com/vllm-project/recipes/blob/main/models/Google/diffusiongemma-26B-A4B-it.yaml)
- [Pinned Jev structured server](https://github.com/vllm-project/vllm/blob/e9757321527ca1ecd514c07c1418dd2c53da3d19/examples/features/structured_diffusion/structured_server.py)
