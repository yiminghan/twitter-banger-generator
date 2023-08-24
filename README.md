# twitter-banger-generator

Imagine the power of your favorite Twitter banger, in the palm of your hands.

!["power of Twitter bangers, in the palm of my hands](banner.jpeg)

Well, now you can!

With Twitter Banger Generator, you can scrap, train, and generate your own Twitter bangers.

Scrap, train, and generate your own Twitter bangers, all locally on your own machine!

## Features

- [x] Automated Builtin Twitter Scraper
- [x] Character-level Transformer training
- [x] GPT2 full fine-tuning (using [nanoGPT](https://github.com/karpathy/nanoGPT))
- [ ] GPT2 qlora fine-tuning (Coming Soon)
- [ ] Llama fine-tuning

## Quick Start

Let's start by generating your banger in a few easy steps!

```bash
# init nanoGPT
git submodule update --init --recursive
# install dependencies
npm install
pip install torch numpy transformers datasets tiktoken wandb tqdm argparse
```

Download pre-trained models:

```
wget https://huggingface.co/yiminghan/twitter-bangers/resolve/main/goth600.pt
mkdir -p nanoGPT/out-goth600
mv goth600.pt nanoGPT/out-goth600/ckpt.pt
```

Find all the models here: https://huggingface.co/yiminghan/twitter-bangers/tree/main

That's it! Let's Generate!

(Honestly, I find it strugging on Mac OS, so I recommend running it on CUDA if you can)

```bash
bash sample_finetune.sh goth600
# On MAC OS, you might need set MPS flag:
MPS=1 bash sample_finetune.sh goth600
```

This generates a lot of stuff, for example:

**Protip: I usually like to generate a lot and pick out the good stuff**

```
% MPS=1 bash sample_finetune.sh goth600
---------------


The concept is simple, if you're not paying attention, you're not paying attention.

It's a game.

It's a living.

It's a time.

It's an obsession.

And you need to be here.

We can start by building a simulation.

We can start by building a simulation.

We can start by creating an AI simulation.

We can start by creating an AI simulation.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
We can start by creating a simulation using only data.
// .....
---------------

```

You can also prompt the beginning of the text with a custom prompt:

```bash
bash sample_finetune.sh goth600 "It's 2023"
MPS=1 bash sample_finetune.sh goth600 "It's 2023"
```

And there's bangers for your specific prompt!

```
---------------
It's 2023. You have internet access. You have data. You have a choice. Have a drink…
— Kevin Spacey, House of Cards
If you have internet access, there's nothing stopping you from connecting to the internet and inventing some new future. No one has the authority to stop you. No one has the right to control you. Just relax and take your time. It's 2023. You have internet access. You have data. You have a choice. Have a drink…
---------------
It's 2023. You have internet access. You are Rich. You are Me.

I am my father's best friend. I am my father's best friend.

My father is my godfather. I am my father's godfather.

You are always there to support me. You are always there to support me.

You are the best. You are the best. You are the best.

You are the best. You are the best. You are the best
---------------

```

## Scrape your own favorite Twitter banger account

All scraping logic is done inside `scraper.spec.js` via [playwright](https://playwright.dev/)

To do begin scraping your favoirte Twitter account, do the following:

1. Set up a throwaway Twitter account to login with, use a temporary email service provider.

2. On line 7 and line 8, replace the `username` and `password` with your throwaway account's credentials.

3. On line 24, replace the `twitterAt` with the Twitter handle of the account you want to scrape.

4. Begin scraping by running `npx playwright test --project=LocalChrome --ui`, and start scraping in the browser window that pops up.

5. There's already some pre-scraped data in `scrap/`, so you can skip the scraping step if you want to just start playing around.

6. Please don't abuse this, I don't want Elon to get pissed.

## Finetuning your own model

Before you finetune, please `cd` into nanoGPT and complete the setup instructions there.

Finetuning is all done via nanoGPT, but there is simple script to do so:

```bash
bash finetune_gpt2.sh {TWITTER HANDLE}
```

(Honestly, I don't recommend finetuning on Mac OS, I recommend running it on CUDA if you can, MacOS is just too slow)

Feel free to play around the learning rates in `finetune.py`

## TODO

**Once I get a good GPU, it's over for you guys. I'm going to train a Llama2 finetune and unleash it on the world.**

But anyways, right now I'm still learning about parameter efficient finetuning techniques, so I plan to add qlora training soon, either submit a PR into NanoGPT or do it in this repo directly.

- [ ] GPT2 qlora fine-tuning (Coming Soon)
- [ ] Llama fine-tuning

## Contributing

PRs Welcome! My GPU is trash right now so I can't train anything bigger, I would be excited to see what a bigger model can generate!
