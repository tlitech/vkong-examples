#!/usr/bin/env python3
"""Minimal Jev server using vLLM's DiffusionGemma logprob reads."""

import argparse
import json
import math
import os
import random
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer


# DiffusionGemma constants used by vLLM's pinned reference implementation.
VOCAB_SIZE = 262144
TURN_CLOSE = 106
PAD = 0
CANVAS_LENGTH = 64
CANVAS_STEP = 16
TOP_LOGPROBS = 20
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"


def parse_questions(body):
    raw = body.get("questions")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("questions must be a non-empty object")
    if len(raw) > 10:
        raise ValueError("this example supports at most 10 questions per request")

    questions = []
    for question_id, value in raw.items():
        if not question_id or ":" in question_id or "\n" in question_id:
            raise ValueError(f"invalid question id: {question_id!r}")
        kind = value.get("type")
        criteria = value.get("criteria") or {}
        if kind == "noul":
            choices = [("yes", criteria.get("true")), ("no", criteria.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice" and isinstance(criteria, dict) and len(criteria) >= 2:
            choices = [(str(name), description) for name, description in criteria.items()]
            if len(choices) > 26:
                raise ValueError(f"question {question_id!r} has more than 26 choices")
            labels = [chr(ord("A") + index) for index in range(len(choices))]
        else:
            raise ValueError(f"question {question_id!r} must be noul or choice")
        questions.append(
            {
                "id": question_id,
                "type": kind,
                "instructions": str(value.get("instructions", "")),
                "choices": choices,
                "labels": labels,
            }
        )
    return questions


def system_prompt(questions):
    text = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    for question in questions:
        text += f"\nQuestion {question['id']}: {question['instructions']}\n"
        for (name, description), label in zip(question["choices"], question["labels"]):
            detail = f" ({description})" if description else ""
            text += f"  {label}: {name}{detail}\n"
    return text + '\nReply with one line per question formatted as "id: label".'


def answer_text(questions, selected_labels):
    return "\n".join(
        f"{question['id']}: {question['labels'][selected]}"
        for question, selected in zip(questions, selected_labels)
    )


def encode(text):
    return TOKENIZER.encode(text, add_special_tokens=False)


def resolve_slots(questions):
    """Find the one token position changed by each question's labels."""
    selected = [0] * len(questions)
    template = SCAFFOLD + encode(answer_text(questions, selected))
    if len(template) + 1 > CANVAS_LENGTH:
        raise ValueError("answer template does not fit the 64-token canvas")

    slots = []
    for question_index, question in enumerate(questions):
        position = None
        label_ids = [0] * len(question["labels"])
        for label_index in range(1, len(question["labels"])):
            candidate_labels = list(selected)
            candidate_labels[question_index] = label_index
            candidate = SCAFFOLD + encode(answer_text(questions, candidate_labels))
            if len(candidate) != len(template):
                raise ValueError(f"labels for {question['id']!r} must each be one token")
            differences = [i for i, token in enumerate(candidate) if token != template[i]]
            if len(differences) != 1 or (
                position is not None and differences[0] != position
            ):
                raise ValueError(f"labels for {question['id']!r} must share one slot")
            position = differences[0]
            label_ids[label_index] = candidate[position]
        label_ids[0] = template[position]
        if len(set(label_ids)) != len(label_ids):
            raise ValueError(f"labels for {question['id']!r} must use distinct tokens")
        slots.append({"position": position, "label_ids": label_ids})
    return template, slots


def canvas_width(template):
    required = len(template) + 1
    return min(CANVAS_LENGTH, -(-required // CANVAS_STEP) * CANVAS_STEP)


def build_canvas(template, slots, seed):
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (canvas_width(template) - len(canvas))
    for slot in slots:
        canvas[slot["position"]] = rng.randrange(VOCAB_SIZE)
    return canvas


def pin_non_answer_tokens(template, slots, steps):
    if steps <= 1:
        return {}
    answer_positions = {slot["position"] for slot in slots}
    return {
        "diffusion_pinned": [
            position
            for position in range(canvas_width(template))
            if position not in answer_positions
        ]
    }


def normalize_logprobs(returned, label_ids):
    """Same label softmax used by vLLM's Jev reference server."""
    floor = min(returned.values()) - 5.0
    label_logprobs = [returned.get(token_id, floor) for token_id in label_ids]
    maximum = max(label_logprobs)
    weights = [math.exp(value - maximum) for value in label_logprobs]
    return [weight / sum(weights) for weight in weights]


def one_read(questions, state, template, slots, seed, steps):
    label_ids = sorted({token_id for slot in slots for token_id in slot["label_ids"]})
    request_body = {
        "model": ARGS.model,
        "messages": [
            {"role": "system", "content": system_prompt(questions)},
            {"role": "user", "content": state},
        ],
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "logprob_token_ids": label_ids,
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": {
            "diffusion_seed_canvas": build_canvas(template, slots, seed),
            "diffusion_canvas_length": canvas_width(template),
            "diffusion_max_steps": steps,
            "diffusion_read_only": True,
            **pin_non_answer_tokens(template, slots, steps),
        },
    }
    request = urllib.request.Request(
        f"{ARGS.upstream}/v1/chat/completions",
        data=json.dumps(request_body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)

    content = result["choices"][0]["logprobs"]["content"]
    distributions = []
    for slot in slots:
        returned = {
            int(item["token"].split(":")[1]): item["logprob"]
            for item in content[slot["position"]]["top_logprobs"]
        }
        distributions.append(normalize_logprobs(returned, slot["label_ids"]))
    return distributions, result.get("usage", {})


def decide(body):
    questions = parse_questions(body)
    state = body.get("state")
    if state is None:
        raise ValueError("state is required")
    state_text = state if isinstance(state, str) else json.dumps(state)
    samples = max(1, min(int(body.get("samples", 2)), 8))
    steps = max(1, min(int(body.get("steps", 1)), 8))
    seed = int(body.get("seed", 42))
    template, slots = resolve_slots(questions)

    started = time.time()
    reads = [
        one_read(questions, state_text, template, slots, seed + index * 7919, steps)
        for index in range(samples)
    ]
    answers = {}
    for question_index, question in enumerate(questions):
        mean = [
            sum(read[0][question_index][label] for read in reads) / samples
            for label in range(len(question["labels"]))
        ]
        winner = max(range(len(mean)), key=mean.__getitem__)
        if question["type"] == "noul":
            answers[question["id"]] = {"type": "noul", "noul": mean[0]}
        else:
            answers[question["id"]] = {
                "type": "choice",
                "choice": question["choices"][winner][0],
                "probabilities": {
                    choice[0]: probability
                    for choice, probability in zip(question["choices"], mean)
                },
                "confidence": mean[winner],
            }
    usage = reads[0][1] if reads else {}
    return {
        "model": ARGS.model,
        "answers": answers,
        "usage": usage,
        "diagnostics": {
            "samples": samples,
            "steps": steps,
            "total_ms": (time.time() - started) * 1000,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_json(200, {"status": "ok"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self.send_json(404, {"error": "not found"})
        if API_KEY and self.headers.get("authorization") != f"Bearer {API_KEY}":
            return self.send_json(401, {"error": "unauthorized"})
        try:
            length = int(self.headers.get("content-length", "0"))
            self.send_json(200, decide(json.loads(self.rfile.read(length))))
        except Exception as error:
            self.send_json(422, {"error": str(error)})


parser = argparse.ArgumentParser()
parser.add_argument("--upstream", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="dgemma")
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--tokenizer-revision")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8011)
ARGS = parser.parse_args()
API_KEY = os.environ.get("API_KEY")
TOKENIZER = AutoTokenizer.from_pretrained(
    ARGS.tokenizer,
    revision=ARGS.tokenizer_revision,
)
SCAFFOLD = encode(SCAFFOLD_TEXT)

print(f"Jev API on {ARGS.host}:{ARGS.port} -> {ARGS.upstream}", flush=True)
ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
