# Presenter script — read this once before the talk

The slides contain very little text. Read the short script below in your own words. The talk takes about 5–7 minutes. You do not need to explain the code.

## Slide 1 — Reading Room

“Good morning. Our project is Reading Room, a computer vision study of pneumonia classification from children's chest X-rays. The four team members are Samarth Shekhar, Meghavi Rao, Rachit Kumar and Harshal Nerpagar. We first reconstructed a published approach, then tested lung-focused cropping and a second type of image model.”

## Slide 2 — Two dataset labels

“The computer predicts one of two labels: normal or pneumonia. These are labels in a public research dataset. The pictures shown here are examples. We evaluate how often predictions agree with the dataset labels. This is a research demonstration, rather than a clinical diagnosis system.”

## Slide 3 — Our journey

“M1 was our literature review. M2 built a baseline, which means a starting system for comparison. M3 recorded our improvement idea before running it. M4 tested that idea, and M5 analyzed the results and built the demonstration. We tried three baseline protocols. The 15-epoch attempt did worse, so we audited the implementation instead of assuming that more training always helps.”

## Slide 4 — Lung-focused input

“First, a pretrained anatomical model estimates where the lungs are. The white mask represents the lungs, not pneumonia lesions. We crop a rectangle around both lungs with a small margin. Then two pretrained image readers classify the image: a CNN called ResNet18 and a vision transformer called DeiT-Tiny. We trained their small final classifier heads for five epochs. The main image encoders stayed fixed.”

## Slide 5 — Four comparisons

“We used the same split and head-training settings in these four conditions. A uses a full image and the CNN. B changes the input to a lung crop. C adds the transformer to the full image. D combines the lung crop with both models. B versus A tests cropping, and D versus B tests the extra transformer on crops. This helps us explain which changes mattered.”

## Slide 6 — Final results

“The final crop-plus-transformer model correctly classified 93.75 percent of 624 test images. It detected 383 of the 390 pneumonia cases, giving 98.21 percent recall. It still made 32 false alarms and missed seven pneumonia cases. Against the matched full-image CNN, accuracy increased by 6.89 percentage points. That comparison is different from comparing against our older five-CNN experiment.”

## Slide 7 — Mistakes and limits

“These are real wrong predictions. A false alarm means a normal-labelled image was predicted as pneumonia. A missed case means a pneumonia-labelled image was predicted as normal. We also found that cropping helped the test set but hurt this validation split. We cannot explain that reversal from one split, so we report it and recommend repeated splits and external testing.”

## Slide 8 — Live demonstration

“We can select a supplied sample or upload a PNG or JPEG. Clicking Analyze runs the saved models. The page shows the original image, predicted lung mask, crop and result. The built-in samples reuse their prepared masks, but classifier inference runs now. Uploaded images run segmentation too. If the network or laptop setup fails, we can show our clearly labelled recorded offline demo.”

# The live demo — exact sequence

1. Before leaving home, download **Presentation_Kit.zip** from GitHub Releases and extract it.
2. If you want live inference, install Python **3.13** and run **Start_Windows.bat** or **Start_Mac.command** from the downloaded repository. The first setup downloads libraries and a 326 MiB model once.
3. Keep the terminal window open. The launcher opens the local browser automatically.
4. Click the first sample. Click **Analyze image**. Wait for the result.
5. Point to the three pictures: **full image → predicted lung mask → crop**.
6. Point to the prediction. Say “This is the model's output.” The percentage is an uncalibrated model score.
7. Select a pneumonia-labelled sample and analyze again. Show the known label and match/mismatch if displayed.
8. Optional: open **Compare the four conditions** or the measured results section.
9. If anything stops working, open **Offline_Demo.html** from the presentation kit. Say: “These are recorded responses from actual model runs.”

# Short answers to likely questions

| Question                                       | Safe, simple answer                                                                                                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Did you train a new model from scratch?        | No. We used pretrained image encoders and trained four small classifier heads for five epochs each. Earlier M2 work also trained specified late CNN layers.          |
| What is an epoch?                              | One pass through the training examples.                                                                                                                              |
| What is a CNN?                                 | An image model that builds patterns from local image regions.                                                                                                        |
| What is a transformer?                         | An image model that relates image patches using attention.                                                                                                           |
| Why crop?                                      | To devote more of the fixed image input to the lungs. It can also remove useful information, so we tested it.                                                        |
| What is segmentation?                          | Predicting which pixels belong to an anatomical region. Here it estimates the lungs.                                                                                 |
| Did you label lung masks?                      | No. We used a fixed pretrained segmenter. We cannot report mask accuracy without manual mask ground truth.                                                           |
| What does 93.75% mean?                         | 585 of 624 benchmark test records were classified correctly.                                                                                                         |
| What does 98.21% recall mean?                  | 383 of 390 pneumonia-labelled test records were found.                                                                                                               |
| How well did it recognize normal cases?        | 202 of 234, or 86.32% specificity. There were 32 false alarms.                                                                                                       |
| Did you beat your base paper?                  | No. Chouhan's paper reports 96.39% accuracy. Our final pilot reports 93.75% under a shorter and different protocol.                                                  |
| Are you best in industry?                      | We have not established that. Published studies report higher scores using different data splits and methods.                                                        |
| Why are validation and test results different? | We do not know the cause from this one split. It shows that the result depends on the partition and needs more validation.                                           |
| Did you pick the model using test accuracy?    | D was the proposed final condition. Checkpoints and blend weights were selected within each condition using validation macro F1. We report every declared condition. |
| What does “macro F1” mean?                     | It averages the F1 scores of normal and pneumonia equally, so a weak normal class is harder to hide.                                                                 |
| What is AUC?                                   | How well the scores rank pneumonia above normal over different thresholds. It is not probability calibration.                                                        |
| Does the white mask show disease?              | No. It is the estimated anatomical lung region.                                                                                                                      |
| What happens if segmentation is poor?          | Empty or implausibly small/large masks use the full-image fallback. This happened for 82 of 5,824 unique images.                                                     |
| Why 256-pixel segmentation?                    | The official 512-pixel setting was too slow for this CPU budget. We recorded the 256-pixel departure and checked sample masks.                                       |
| What is actually trained in final D?           | Its two linear heads have 1,412 trainable parameters. The fixed encoders and segmenter are much larger; the complete pipeline has about 84.77 million parameters.    |
| What would you do next?                        | Repeat group-aware splits, test an independent dataset, and evaluate limited backbone fine-tuning and better pediatric segmentation.                                 |

# Four claims to avoid

- “We trained the whole transformer from scratch.”
- “The mask identifies pneumonia lesions.”
- “The score is the patient's actual probability of pneumonia.”
- “We beat every paper or proved clinical readiness.”

The full report contains all three baseline attempts, the four final conditions, graphs, failure examples, exact settings and limitations.
