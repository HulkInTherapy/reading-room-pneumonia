# Samarth Shekhar — read this before tomorrow

**Roll no:** 230042 · **You present:** slide 3, *Foundations* · **Your time:** ~2 minutes

> This is the only file you need. You do not have to open the slides, read any paper,
> or look anything up. Read this once (~15 min), and you can walk in and talk.
> Everything is explained from scratch — no jargon is used before it is defined.

---

## 0 · If you only remember four sentences

1. We are reviewing 12 papers on **detecting pneumonia in children's chest X-rays**
   using AI, on a public dataset of 5,856 images from one hospital in Guangzhou, China.
2. Over six years, reported accuracy on that dataset climbed from **92.8% to 98%** — but
   the test set is only **624 images** and the validation set is **16 images**.
3. **Our research gap:** the papers using this dataset never check whether their models
   actually work — no external hospital, no error bars, no calibration, and their
   explanation heatmaps are shown but never scored. The papers that *do* run those checks
   all use different datasets. **The two literatures never touch.**
4. **You present the three foundation papers** — the dataset, the recipe, and the
   explanation method that everything else in the review is built on.

---

## 1 · What is this project, actually?

It is a **research-reproduction project**. We are not inventing something from nothing.
The assignment has six steps, and the whole course is graded across five milestones:

| # | Step | What it means |
|---|------|---------------|
| 1 | **Literature survey** | Read 10–20 recent papers on our topic. ← **this is tomorrow (M1)** |
| 2 | Pick a base paper | Choose ONE paper with public code + downloadable data |
| 3 | Reproduce | Re-run their method, land near their reported number |
| 4 | Hypothesis | Propose ONE change of our own, and **predict the result before coding** |
| 5 | Implement & test | Make the change, run ablations, measure what moved |
| 6 | Analyse & write | Failure analysis + write it up like a real IEEE conference paper |

**Tomorrow is step 1 only.** We present, in 10 minutes, what we read and what we found.

There is one extra instruction that matters more than the rest. The exact words were:

> *"Bonus points if you can find a problem/shortcoming still present in all of them
> which none of the papers have solved (this is called a research gap)."*

**We found one.** That is the strongest part of our presentation, and every one of the
four of us should understand it, because that is what she is most likely to ask about.

Our chosen topic (already submitted): **Pneumonia detection from chest X-rays,
using the Kermany CXR dataset.**

---

## 2 · The medical problem, in plain words

**Pneumonia** is an infection that inflames the tiny air sacs in the lungs (alveoli).
They fill with fluid or pus instead of air. It is one of the largest causes of death
in children under five worldwide.

**How it is diagnosed:** a chest X-ray. Healthy lungs are mostly air, and air lets
X-rays pass straight through, so healthy lungs look **dark** on the film. When the air
sacs fill with fluid, that region blocks more X-rays and shows up as a **white or hazy
patch** — radiologists call this an *opacity* or *consolidation*.

**So why is this hard?**

1. The patch is often faint, and can hide behind the heart or the ribs.
2. Other conditions look similar on film.
3. In children it is even harder — smaller anatomy, and kids move during the scan.
4. Most importantly, **there are not enough radiologists.** In many parts of the world
   an X-ray gets taken and then waits hours or days for someone qualified to read it.
   That delay is where children die.

So the goal is **not** to replace a doctor. The goal is a **triage tool** — something
that flags the urgent films so a human looks at those first.

**One more distinction that comes up constantly:** pneumonia can be **bacterial** or
**viral**. This matters enormously because bacterial pneumonia needs antibiotics and
viral pneumonia does not. Telling them apart on an X-ray is much harder than just
detecting that *something* is wrong — and you will see every paper score worse on it.

---

## 3 · Where computer vision comes in

The task we are doing is **image classification**: input one X-ray, output one label.

Everything in this field is built from four ideas. Learn these four and you can follow
any of the twelve papers.

### 3.1 · CNN — Convolutional Neural Network

The workhorse. Think of it as looking at the image through a small sliding window:

- **Early layers** detect very simple things — edges, corners, light-to-dark transitions.
- **Middle layers** combine those into textures and shapes — rib curves, hazy regions.
- **Late layers** combine *those* into concepts — "a cloudy white area in the lower left lung."
- **Final layer** collapses everything into one number: the probability of pneumonia.

Crucially, **nobody tells it what to look for.** You show it thousands of labelled images,
it makes guesses, you tell it how wrong it was, and it nudges its internal numbers
(*weights*) to be less wrong next time. Repeat millions of times. It works out its own
features. This is why the "what is it actually using?" question is so hard — and why
it is the heart of our project.

### 3.2 · Transfer learning

Training a CNN from zero needs millions of images. We have 5,856.

The trick: take a network **already trained on ImageNet** — 1.4 million everyday
photographs of cats, cars, chairs. It has already learned edges, curves, textures, and
how to combine them. Then keep those early layers ("**freeze**" them) and retrain only
the last few layers on X-rays. This is called **fine-tuning**.

You get a strong model from a few thousand images instead of a few million.
**Almost every paper we read does exactly this.** The differences between papers are
mostly just *which* pretrained network they grabbed.

The networks you will hear named — all of them are just different CNN designs:

| Name | The one thing to know |
|------|----------------------|
| **VGG16** | Old and simple: a plain deep stack of small 3×3 filters. Sturdy, heavy. |
| **Inception-V3** | Looks at several filter sizes in parallel at each layer, then merges. |
| **ResNet** | Adds *skip connections* so a layer can pass its input forward unchanged. This is what let networks get really deep without training collapsing. |
| **DenseNet** | Takes that further — every layer receives input from *all* previous layers. Best performer on this task in several papers. |
| **Xception** | Splits each convolution into two cheaper steps (space first, then channels). Efficient. |
| **ViT** (Vision Transformer) | Not a CNN at all — explained in §9 if it is on your slide. |

### 3.3 · Data augmentation

We only have 5,216 training images, and a big network can simply **memorise** them
(this is called **overfitting** — perfect on training data, useless on new data).

Fix: make fake-but-realistic extra copies. Rotate slightly, zoom, shift, flip, adjust
brightness. The lungs are still lungs, the label is still correct, but the network can
no longer memorise exact pixels — it has to learn the actual pattern.

### 3.4 · Grad-CAM — the "explanation" heatmap

A coloured heatmap laid over the X-ray, showing which pixels most influenced the answer.
Red = mattered a lot. Blue = did not matter.

It works by taking the last convolutional layer's feature maps and weighting them by how
much each one pushed the final "pneumonia" score up.

**This is the standard way papers claim "our model looks at the right place."**
Remember that phrasing — *claim*. Whether that claim survives being measured is a
central part of our story.

---

## 4 · Our dataset — know these numbers cold

This is the **Kermany chest X-ray dataset** (also on Kaggle as "Chest X-Ray Images
(Pneumonia)"). Released alongside the Kermany 2018 paper.

| Split | Images |
|-------|--------|
| Train | **5,216** |
| Test | **624** |
| Validation | **16** |
| **Total** | **5,856** |

Four facts that drive our entire argument:

1. **One hospital.** Guangzhou Women and Children's Medical Center, China.
   One site, one set of machines, one acquisition protocol, one population.
2. **Children aged 1–5 only.** Not adults. Not newborns.
3. **The classes are imbalanced: 4,273 pneumonia vs 1,583 normal**, roughly 2.7 : 1.
   Consequence: a model that stupidly answers "pneumonia" for **every single image**
   scores **62.5%** on the test set without learning anything. So "62.5%" is our floor,
   not 50%. Any accuracy number must be read against that.
4. **The validation split is 16 images.** Eight per class. Read that again.

Why (4) is a genuine scandal and not a nitpick:

> You need three splits. **Train** = what the model learns from. **Validation** = what you
> use to make decisions (when to stop training, which learning rate, which architecture).
> **Test** = touched once, at the very end, to report an honest number.
>
> With 16 validation images you cannot tell a 93% model from an 87% model — the
> difference is one image. So in practice everyone tunes against the **test** set and
> then reports on that same test set. That is called **selecting on the test set**, and it
> makes every published number optimistic by construction.

**And: 624 test images gives a 95% confidence interval of roughly ±1.5 percentage points.**
A "1% improvement" is about six images. Hold on to that number — it is the backbone of
our research gap.

---

## 5 · Jargon decoder

Anything you might need to say or defend. Skim now, come back if she asks.

| Term | Plain meaning |
|------|---------------|
| **Accuracy** | Fraction of all predictions that were correct. Misleading when classes are imbalanced (see the 62.5% floor above). |
| **Sensitivity / Recall** | Of the children who **actually have** pneumonia, what fraction did we catch? **This is the one that matters medically** — a missed sick child is the dangerous error. |
| **Specificity** | Of the **healthy** children, what fraction did we correctly clear? Low specificity = lots of false alarms. |
| **Precision** | Of the ones we *flagged* as pneumonia, what fraction really had it? |
| **F1 score** | A single number balancing precision and recall (their harmonic mean). Used when classes are imbalanced. |
| **AUC / AUROC** | The model outputs a score from 0 to 1. Accuracy forces you to pick a cutoff (say 0.5); AUC measures quality across **every possible cutoff**. Concretely: pick one random sick patient and one random healthy patient — AUC is the probability the model scores the sick one higher. **0.5 = coin flip, 1.0 = perfect.** |
| **Epoch** | One full pass through the training data. |
| **Overfitting** | Model memorises training data; great on train, bad on anything new. |
| **Ablation** | Deliberately remove/disable one component to see how much it was contributing. |
| **Confidence interval (CI)** | The range the true value probably sits in, given you only measured on a limited sample. Small test set → wide interval. |
| **Calibration** | When the model says "90% confident," is it right 90% of the time? Neural networks are famously **overconfident** — they say 99% when they are right 80% of the time. |
| **ECE** | *Expected Calibration Error.* One number summarising that mismatch. Lower is better. Zero papers we read report it. |
| **Segmentation** | Instead of one label for the whole image, label **every pixel** — e.g. "this pixel is lung, this one isn't." Output is a **mask**. |
| **U-Net** | The standard CNN design for segmentation. Shrinks the image down to understand context, then expands it back to full resolution to draw precise borders. |
| **External validation** | Testing on data from a **completely different hospital** than you trained on. The real test of whether a medical model works. |
| **Shortcut learning** | The model finds the **easiest** thing that predicts the label — even if it is medically meaningless. Explained fully in §7. |
| **Domain shift** | The new data looks different from the training data (different scanner, different hospital, different patients), so performance drops. |
| **Baseline** | The number you must beat. Ours is 62.5% (always-say-pneumonia). |

---

## 6 · What your teammates are covering

*You set the table; they build on it. One line each so you know where your part fits and
can follow the rest of the talk.*

**Meghavi — slide 4, the accuracy race.** Three papers that each take your Kermany
dataset and swap in a different pretrained network:
- **Stephen 2019** — builds a CNN from scratch with heavy augmentation, no ImageNet
  weights, and still competes. Suggests the gains aren't really about architecture.
- **Ayan & Ünver 2019** — honest head-to-head: VGG16 87%, Xception 82%. The useful bit is
  that they make *different* mistakes (VGG16 more specific, Xception more sensitive).
- **Rahman 2020** — sweeps four backbones; DenseNet-201 wins at ~98%, the highest headline
  on this benchmark. Drops to 93.3% on bacterial vs viral.

**Rachit — slide 5, pushing past 96%.**
- **Chouhan 2020** — ensembles five networks (including your CheXNet DenseNet lineage):
  96.4% accuracy, 99.62% recall.
- **Rajaraman 2018** — customised VGG16 at 96.2%, plus CAM heatmaps. The closest anyone on
  this dataset gets to your ChestX-ray8 localization idea — but only pictures, never scored.
- **Singh 2024** — a Vision Transformer at 97.61%, the highest number in the review.

**Harshal — slide 6, the reality check.** The papers that take all of it apart:
- **Zech 2018** (*PLOS Medicine*) — AUC 0.931 internal collapses to 0.815 at a new
  hospital; a CNN could identify *which hospital* an X-ray came from with 99.95% accuracy.
- **DeGrave 2021** (*Nature Mach. Intell.*) — names the mechanism: **shortcut learning**.
  Models key on markers, borders and positioning, not lungs.
- **Saporta 2022** (*Nature Mach. Intell.*) — scored seven saliency methods against
  radiologists. Grad-CAM was best of the seven and **still significantly worse than a
  human**. This is the paper that retroactively undermines every Grad-CAM figure in the
  review — including the ones in your CheXNet paper.

---

## 7 · The story of the field, in one page

This is the arc of our whole presentation. If you understand this, you can improvise.

**2017–2018 — the tools arrive.** Public datasets appear (ChestX-ray8, Kermany). Transfer
learning is shown to work on small medical datasets. Grad-CAM becomes the standard way
to "explain" a prediction. Kermany reports **92.8%**.

**2018–2024 — the race.** Everyone downloads the same 5,856 images and swaps in a
different pretrained network. Reported accuracy climbs: 96.2 → 96.4 → 97.6 → **98.0**.
Ensembles appear. Vision Transformers appear. Every paper reports a higher number than
the last, on **the same 624 test images**.

**Meanwhile, in a completely separate literature — the reality check.** A different set
of researchers, publishing in *PLOS Medicine* and *Nature Machine Intelligence*, asked
whether models like these actually work. They found:

- Move the model to another hospital and performance **collapses** (Zech).
- The models are often keying on **things outside the lungs entirely** — equipment
  markers, image borders, how the patient was positioned (DeGrave). This is
  **shortcut learning**: a neural network finds the *easiest* feature that predicts the
  label, and if a shortcut exists in your data it will take it, because it is cheaper to
  learn than actual disease.
- And the Grad-CAM heatmaps everyone uses as proof? When you **score** them against a
  radiologist's own markings, they are **significantly worse than a human** (Saporta).

**And here is the punchline of our entire presentation:**

> **These two literatures never touch.**
>
> Every paper that uses our dataset reports one accuracy number and never checks any of
> this. Every paper that does the checking uses a *different* dataset.
> Nobody has run the reality check on this benchmark.

---

## 8 · THE RESEARCH GAP — the bonus-marks part

**Say it like this:**

> "Every paper on this benchmark reports a single accuracy number on the same 624
> single-centre images. Every paper that tested whether such models actually work used a
> different dataset. Nobody has run the reality check on the benchmark itself."

It breaks into five specific, checkable gaps. Not one of the twelve papers closes them:

| # | Gap | Why it matters |
|---|-----|----------------|
| **1** | **Selection on the test set** | The validation split is 16 images, so it cannot rank models. Tuning happens against the test set. Reported numbers are optimistic **by construction**. |
| **2** | **No error bars** | Zero of the nine benchmark papers report a confidence interval or run multiple random seeds. |
| **3** | **No external validation** | One hospital, one city, ages 1–5. Zech measured a **−0.12 AUC** drop off-site elsewhere. Nobody has measured it here. |
| **4** | **No calibration** | Not one paper reports ECE or a reliability curve. A triage tool needs a *trustworthy confidence*, not just a correct label. |
| **5** | **Heatmaps never scored** | Grad-CAM appears as figures in most papers. Saporta showed that actually scoring them **changes the verdict**. Nobody scores them here. |

### The statistics point — get this exactly right

There is a subtle version of gap 2 that is worth stating carefully, because getting it
half-right is worse than not saying it.

624 test images → the 95% confidence interval is about **±1.5 percentage points**.
Now lay the reported numbers against that window:

- Kermany 2018 → **92.8%**
- Rajaraman 2018 → **96.2%**
- Chouhan 2020 → **96.4%**
- Singh (ViT) 2024 → **97.6%**
- Rahman 2020 → **98.0%**

**The first jump is real.** 92.8 → 96.2 clears the ±1.5 window comfortably.

**Everything after that is not separable from noise.** 96.2, 96.4, 97.6 and 98.0 all sit
inside overlapping windows. So roughly **five years of published "improvement" cannot be
distinguished from random variation** — and because nobody reports a CI or a seed
variance, none of those authors could have known either way.

⚠️ **Do not say "all the gains are noise."** That is wrong and she may catch it.
Say: *"the first jump is real; every gain claimed since 2019 sits inside the noise window."*

---

## 9 · What we are actually going to build (milestones 2–5)

### Base paper: Kermany et al. 2018 (Inception-V3)

Chosen for **reproducibility, not performance**. It has public data, a clearly stated
architecture, and three clearly stated targets to land near: **92.8% / 93.2% / 90.1%**.
Small enough to train on a free GPU.
*(Fallback if reproduction stalls: Rahman 2020, DenseNet-201 — same data, more
implementations floating around.)*

### Our ONE change: train and test on **lung-cropped** X-rays

The assignment allows exactly one change of our own. Here it is:

1. Run a **pretrained lung segmentation model** (a U-Net trained on the public Montgomery
   and Shenzhen chest X-ray sets) over every image. It outputs a **mask** marking which
   pixels are lung.
2. **Crop to that region.** Everything the shortcut literature blames — laterality
   markers, collimation edges, image borders, burnt-in text, shoulders, positioning
   cues — is now *physically gone from the input*.
3. Train **two identical models**: one on full images, one on cropped. Same architecture,
   same hyperparameters, same seeds. **Only the input region changes.**

### Our prediction — submitted BEFORE we write any code

The rules say: *predict first, being wrong is fine, good reasoning is what counts.*

- **On the normal test set:** the two models will land within about **1 point** of each
  other. Cropping adds no information — the lungs already carry the real signal.
- **On a context-ablation stress test** — invert the mask, so we **black out the lungs**
  and keep only the surrounding context — the full-image model will **still score well
  above the 62.5% baseline**, because it can lean on shortcuts. The cropped model, having
  never seen that context, should **fall toward chance**.
- **That gap is the amount of shortcut, measured as a number.** Which is exactly what
  nobody has done on this dataset.
- Secondary predictions: the cropped model should be **better calibrated** (lower ECE)
  and have a **higher Grad-CAM hit rate inside the lung field**.

### And if the prediction is wrong?

Still a result, and the brief explicitly allows it. If cropping costs several accuracy
points, that means genuine clinical signal lives outside our mask — most likely the
**costophrenic angles** (the sharp corners where the diaphragm meets the ribcage, where
fluid pools) and the **retrocardiac region** (behind the heart). Segmentation models
routinely clip both. We would report that as a real finding about the segmentation,
and it goes straight into our limitations section.

### The honest-evaluation harness around it

This part is arguably as valuable as the change itself, because it is what the field skipped:

- **Patient-wise re-split** — all images from one child stay in one split. Otherwise the
  same patient appears in train *and* test and you are measuring memorisation. Gives us a
  real validation set instead of 16 images.
- **5 random seeds**, reporting **mean ± 95% CI** on every number.
- **Sensitivity/recall + AUC**, never accuracy alone (this is also a course rule).
- **ECE + a reliability diagram** — would be the first calibration numbers published on
  this benchmark.
- **Grad-CAM hit rate scored inside the lung mask** — measured, not displayed.

---


## 10 · YOUR THREE PAPERS — in depth

You have the **foundations**. Everything the other three present is descended from one of
your papers, so you are setting the table for the whole talk. Your job is to establish
three things: *here is the dataset, here is the recipe, here is how people claim to
explain these models.*

---

### P1 · Kermany et al., 2018, *Cell*
**"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"**
~6,100 citations · **this is our chosen base paper**

**The idea they are proposing.** Before this, the assumption was that medical AI needed a
huge purpose-built medical dataset and a custom architecture. Kermany's argument was the
opposite: *you don't need either.* Take a network already trained on ordinary photographs,
freeze the part that has learned generic vision (edges, textures, shapes), and retrain
only the top layers on medical images.

**What they actually did.** Used **Inception-V3** pretrained on ImageNet. Initialised the
convolutional layers with those pretrained weights, froze them, retrained the classifier
head. Their main subject was retinal OCT scans; then, to prove the approach generalises,
they applied the same recipe to **pediatric chest X-rays** — 5,856 images they collected
from Guangzhou Women and Children's Medical Center.

**Results:**
- Normal vs pneumonia: **92.8% accuracy, 93.2% sensitivity, 90.1% specificity**
- Bacterial vs viral pneumonia: **90.7% accuracy**

**Why it mattered so much.** Two reasons. First, it proved transfer learning works in
medicine with only a few thousand images — that unlocked the whole field for people
without hospital-scale data. Second, and honestly more important for us: **they released
the dataset publicly.** That is why it became *the* pneumonia benchmark, and why eight of
our twelve papers use it.

**What is missing.** One accuracy number, one test set, one hospital. No external
validation, no confidence intervals, no calibration. They show occlusion maps
(a heatmap-style visualisation) but only as illustrations — never scored.

**Why we chose it as our base paper.** Not because it is the best — it is the *lowest*
number of the nine. We chose it for **reproducibility**: public data, a plainly stated
architecture, and three specific numbers to land near. You cannot reproduce a paper that
doesn't tell you what it did.

---

### P2 · Rajpurkar et al., 2017 — **CheXNet** (Stanford ML Group)
**"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"**

**The idea they are proposing.** Where Kermany went *efficient*, Stanford went *big*.
Their bet: with a large enough dataset and a deep enough network, you can match a
practising radiologist.

**What they did.** A **121-layer DenseNet** trained on **ChestX-ray14** — 112,120 frontal
chest X-rays — predicting **14 diseases simultaneously** (multi-label, not binary).
Trained with Adam, batch size 16, learning rate decayed when validation plateaued.

*DenseNet in one sentence:* every layer receives input from **all** previous layers, so
information and gradients flow freely, letting you go very deep without the training
signal dying out.

**The headline result.** On pneumonia specifically: **F1 = 0.435** (95% CI 0.387–0.481)
versus four practising radiologists averaging **F1 = 0.387** (95% CI 0.330–0.442).
Beating the humans is what generated the "radiologist-level" headline.

**⚠️ The number you must present honestly.** **0.435 is a low F1.** If someone asks
"is this solved?", this is your evidence that it is not. Both the AI and the humans score
poorly, because detecting pneumonia on a single frontal film is genuinely hard.

**A fair criticism worth knowing.** The radiologists were shown only the image — no
clinical history, no lateral view, no prior films. Real radiologists have all of that.
So it is not quite an apples-to-apples comparison.

**What it gave the field.** Two things that stuck: **DenseNet as the default backbone**,
and **Grad-CAM heatmaps as the standard "explanation."** Almost every later paper copies
both. It is also one of the very few papers in our review that reported **bootstrap 95%
confidence intervals** — remember that when we get to the gap slide, because it makes the
point that the field *knew how* to do this and simply stopped bothering.

---

### P3 · Wang et al., 2017, CVPR — **ChestX-ray8** (NIH)
**"Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification
and Localization of Common Thorax Diseases"**

**The idea they are proposing.** The bottleneck in medical AI is not models, it is
**labelled data** — and the reason there is no labelled data is that expert labelling
costs a fortune. Their proposal is a way around that.

**The trick.** Hospitals already have, for every X-ray, a **written radiology report**
by the doctor who read it. So don't pay anyone to re-label images — run **NLP** over
those existing text reports and mine the disease labels automatically. That gave them
**108,948 X-rays from 32,717 patients** labelled with 8 conditions, essentially for free.

**The catch — and say this out loud, it is a good point.** The labels describe *the
report*, not the image. NLP misreads negations and uncertainty ("no evidence of
pneumonia" is easy to get backwards). So the labels are **noisy**, which puts a ceiling
on how well *anything* trained on this data can score. That partly explains why CheXNet's
F1 of 0.435 looks so low next to the 98% figures on our cleaner dataset.

**The second contribution — the one that matters most for our project.**
"**Weakly-supervised localization.**" They had image-level labels ("this X-ray shows
pneumonia") but not pixel-level ones, and they asked whether a model trained only on
those weak labels could still point to *where* the disease is. They hand-annotated
**bounding boxes** on a subset and **scored** the model's attention against them at
various overlap thresholds.

**Why that is the paper's real legacy for us.** It is the one early paper that treated
"is the model looking in the right place?" as **a number to be measured**, not a
picture to be admired. Everything on Harshal's slide, and our own proposed experiment,
descends from that idea.

---

### How to hand off to Meghavi

End with something like:

> *"So by 2018 the field has everything it needs: a public dataset, a recipe —
> pretrained CNN, fine-tune the top — and a standard way to draw explanations.
> Now watch what everyone does with it."*


---

## 11 · Your ~2 minutes — a script you can just say

*Not to be memorised word for word. Say it in your own words; this is the shape.*

> *"I'm covering the three papers everything else is built on.*
>
> ***Kermany 2018, in Cell.*** *Their idea was that you don't need a huge medical dataset
> or a custom network — take Inception-V3, already trained on 1.4 million ordinary
> photographs, freeze the layers that learned generic vision, and retrain just the top on
> X-rays. They got 92.8% accuracy, 93.2% sensitivity, 90.1% specificity on pediatric
> chest X-rays. Two reasons this paper matters: it proved transfer learning works in
> medicine with small data, and they released the dataset — which is why eight of our
> twelve papers use it. It's also the paper we've chosen to reproduce, because it's
> public and clearly specified.*
>
> ***CheXNet, Stanford, 2017.*** *The opposite bet — go big. A 121-layer DenseNet on
> 112,000 adult X-rays, 14 diseases at once. It got an F1 of 0.435 on pneumonia against
> four radiologists averaging 0.387, which is where 'radiologist-level' came from. But
> I'd flag that 0.435 is a low score in absolute terms — both the AI and the humans do
> badly, because pneumonia is genuinely hard. This paper is why everyone after it uses
> DenseNet and Grad-CAM heatmaps.*
>
> ***ChestX-ray8, NIH, CVPR 2017.*** *108,000 images. They couldn't afford to label
> that many, so they ran NLP over the existing written radiology reports to mine labels
> automatically — cheap, but noisy, which caps performance. Their second contribution is
> the one that matters to us: they added bounding boxes and actually scored whether the
> model looked in the right place. That's the only early paper that measured localization
> instead of just showing a heatmap — and that idea comes back at the end of our talk.*
>
> *So by 2018 the field has a public dataset, a recipe, and a way to draw explanations.
> Meghavi will show you what everyone did with that."*

---

## 12 · If you get asked something

**"Why did you choose Kermany as your base paper and not the 97.6% Vision Transformer?"**
Reproducibility, not performance. Kermany gives us public data, a clearly stated
architecture, and three stated metrics to land near. We need a baseline we can actually
match before we change one thing. It's the *lowest* number of the nine — that's fine,
it's not a leaderboard entry, it's a starting point.

**"What exactly is transfer learning?"**
Training a CNN from scratch needs millions of images and we have 5,856. So take a network
already trained on ImageNet — 1.4 million everyday photos — which has already learned
edges, curves and textures. Freeze those early layers, retrain only the last few on
X-rays. You get a strong model from a few thousand images.

**"Why is CheXNet's F1 only 0.435 if other papers report 97% accuracy?"**
Different dataset and a much harder task. ChestX-ray14 is adult, multi-label,
14 pathologies, with noisy NLP-mined labels. Kermany is pediatric, binary, clean and
single-site — far easier. That difficulty gap is itself evidence that the 98% numbers on
our dataset are flattered by the dataset, not by better models.

**"What does 'weakly-supervised localization' mean?"**
They had labels saying "this X-ray shows pneumonia" but no labels saying *where*.
Weakly-supervised means training only on the whole-image label, then asking whether the
model can still point to the right region. They checked it against hand-drawn bounding
boxes.

**"Why are NLP-mined labels a problem?"**
The label describes the written report, not the image, and NLP handles negation and
uncertainty badly — "no evidence of pneumonia" can easily be mined as positive. So the
labels are maybe 90% right, which caps how well anything trained on them can do.

**The universal escape hatch.** If you genuinely do not know:

> *"We didn't measure that — that's actually one of the things our experiment is set up
> to answer."*

This is a **strong** answer, not a weak one. Our entire presentation is an argument that
this field claims things it has not measured. Refusing to guess is us being consistent.
Never invent a number.

---

## 13 · Cheat card — the numbers worth memorising

| Number | What it is |
|--------|-----------|
| **5,856** | total images (5,216 train / 624 test / **16 val**) |
| **16** | the validation set. The scandal. |
| **624** | test images → **±1.5 pt** confidence interval |
| **2.7 : 1** | class imbalance (4,273 pneumonia / 1,583 normal) |
| **62.5%** | score you get by always answering "pneumonia" — our real floor |
| **1–5 years** | patient ages. One hospital, Guangzhou, China. |
| **92.8 / 93.2 / 90.1** | Kermany's accuracy / sensitivity / specificity — our reproduction target |
| **98.0%** | best reported on this benchmark (Rahman 2020) |
| **0.931 → 0.815** | Zech's AUC, internal → external hospital. The collapse. |
| **99.95%** | accuracy with which a CNN could identify *which hospital* an X-ray came from |
| **F1 0.435 vs 0.387** | CheXNet vs radiologists — *both low*, pneumonia is hard |
| **108,948** | ChestX-ray8 images, labels NLP-mined from reports |

---

## 14 · Running order tomorrow

| Slide | Who | What |
|-------|-----|------|
| 1 | anyone | Title — topic, team |
| 2 | anyone | The dataset: 5,216 / 624 / **16**, one hospital, ages 1–5 |
| 3 | **Samarth** | Foundations — Kermany, CheXNet, ChestX-ray8 |
| 4 | **Meghavi** | The accuracy race — Stephen, Ayan & Ünver, Rahman |
| 5 | **Rachit** | Ensembles, interpretability, transformers — Chouhan, Rajaraman, Singh |
| 6 | **Harshal** | The reality check — Zech, DeGrave, Saporta |
| 7 | anyone | **The key chart** — accuracy climbed, evidence didn't |
| 8 | anyone | **The research gap** — the bonus-marks slide |
| 9 | anyone | What we build next — base paper, our one change, our prediction |

Total 10 minutes. Roughly 2 minutes each for slides 3–6, and the last three slides are
short. If someone does not want to speak, any of the others can take slide 7, 8 or 9 —
they are written to be presentable by anyone.

**One last thing.** If she pushes hard on any single point, the safest ground in the
entire presentation is this sentence:

> *"Every paper here reports one number on one small single-hospital test set, and nobody
> checked whether the model is looking at the lungs. That's our gap, and that's what we're
> going to measure."*

Good luck. You know this.
