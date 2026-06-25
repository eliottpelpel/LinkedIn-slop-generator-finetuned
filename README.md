# LinkedIn Slop LoRA

Fine-tuned Llama 3.2-3B-Instruct to turn your most boring and mundane life events into LinkedIn thought-leadership, B2B sales, assertive and inspiring posts.

## What it does

Give it anything (random fact, thing that happened to you,etc). Get back a post about resilience, growth mindset, and servant leadership.

## Examples

**Prompt:** `I worked until 2am.`

> Base model: *"That sounds like a long and tiring night..."*
>
> **Tuned:** Worked until 2am tonight. Not because I was 'fired up' or 'in flow.' Because sometimes the only way to get it done is to be the only one who can see it's not done. 🛠️📆
>
> That 2am epiphany? Worth it.
>
> Finish what nobody else will. Even if nobody's watching except yourself.
>
> \#LateNightWins #Discipline #FinishWhatYouStart #NightOwlConfessions

---

**Prompt:** `My dog puked on the couch and I had to clean it.`

> Base model: *"Here's a step-by-step guide on how to clean a vomited-on couch..."*
>
> **Tuned:** My dog puked on my couch yesterday. Cleaned it myself. No wipes. No paper towels. Just me, a bottle of white vinegar, and a lot of patience. 🐕🛀
>
> Leaders don't always give speeches. They clean up after their team's messes, one stain at a time.
>
> \#ServantLeadership #Humility #TeamFirst

---

**Prompt:** `The salad I ate at lunch gave me food poisoning.`

> Base model: *"Here are some general tips that might help you feel better..."*
>
> **Tuned:** Got food poisoning by dinner. Not the healthiest way to learn about the consequences of your diet.
>
> But it was a brutal — and necessary — reminder that growth doesn't always feel comfortable. 🥗🤒
>
> What growth project are YOU pushing through despite some potential discomfort?
>
> \#Discomfort #GrowthMindset #LearningThroughPain

## Usage

**Requirements:** access to `meta-llama/Llama-3.2-3B-Instruct` on HuggingFace (request access [here](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)), plus `torch`, `transformers`, and `peft`.

```bash
pip install torch transformers peft
```

Run with any mundane prompt:

```bash
python generate.py "I spilled coffee on my keyboard."
```

The script loads the base model and the fine-tuned adapter, then prints both outputs side by side so you can see what changed.

## How it was built

| Step | Detail |
|------|--------|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Method | LoRA (r=16, α=32) on `q_proj` and `v_proj` |
| Dataset | 117 train / 16 validation prompt→post pairs |
| Training | 15 epochs, lr=2e-4, completion-only loss |
| Hardware | Apple M-series (MPS) |

The dataset pairs simple prompts ("I forgot to defrost chicken for dinner") with the kind of post that frames it as a leadership lesson. The model learns the voice, structure, and hashtag cadence of that genre.

## Files

```
generate.py              # run inference from the command line
linkedinfinetune.ipynb   # training + generation notebook
train.jsonl              # 117 training examples
valid.jsonl              # 16 validation examples
linkedin-slop-lora-final # saved LoRA adapter weights
```
