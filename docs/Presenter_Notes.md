# Presenter script — read this once before the talk

New to the project? First read Friend_20_Minute_Guide.pdf. The slides now include brief explanations; use this script to connect them into a story. The talk takes about 5–7 minutes. You do not need to explain the code.

## Slide 1 — Reading Room

“Good morning. Our project is Reading Room, a computer vision study of pneumonia classification from children's chest X-rays. The four team members are Samarth Shekhar, Meghavi Rao, Rachit Kumar and Harshal Nerpagar. We first reconstructed a published approach, then tested lung-focused cropping and a second type of image model.”

## Slide 2 — Dataset and prediction task

“We use the public Kermany pediatric chest X-ray dataset. It contains normal and pneumonia labels, with 5,856 original records. Our final experiment uses 4,164 training, 1,042 validation and 624 test records. The test set contains 234 normal and 390 pneumonia records. These images come from one pediatric center, so they do not establish performance in every hospital.”

## Slide 3 — Our journey: M1 to M5

“This table shows our entire project. M1 reviewed earlier work. In M2 our first cached run reached 87.18 percent, the longer audit reached 85.74 percent, and corrected online processing reached 92.15 percent. M3 recorded our hypothesis before implementing it. M4 ran four matched comparisons, with final accuracy 93.75 percent. M5 verified the model and analyzed errors. Because historical settings changed together, we use A/B/C/D for the fair final comparison.”

## Slide 4 — From full X-ray to lung crop

“We start with the full image. A fixed pretrained anatomical model estimates both lungs, producing the white mask. White means predicted lung pixels, not infection. We crop a rectangle with a 10 percent margin from the original pixels, then resize it to 224 by 224 in RGB format. RGB is the three-channel input format, not medical coloring. ResNet18 and DeiT-Tiny classify this crop, and final D averages their scores equally.”

## Slide 5 — Four matched comparisons

“A uses the full X-ray and CNN. B changes only to the lung crop, so B versus A tests cropping. C adds the transformer on full images. D adds it on crops, so D versus B tests the extra model there. All conditions use the same split and comparable five-epoch head training. The main encoders remain fixed. These are measured controls, not guesses about which change helps.”

## Slide 6 — Final model results

“Our final model correctly classified 585 of 624 test records, giving 93.75 percent accuracy. It found 383 of 390 pneumonia records, giving 98.21 percent recall. A false alarm is a normal-labelled image predicted as pneumonia: there were 32. A missed case is pneumonia predicted as normal: there were seven. Against the matched full-image CNN control, accuracy increased by 6.89 percentage points.”

## Slide 7 — Mistakes, limits and next steps

“These are two real wrong predictions, showing a false alarm and a missed pneumonia case. A good overall score does not mean every image is right. Cropping helped test performance but reduced validation performance. We cannot establish the reason from one split. Our next work is repeated group-aware splits, independent data and limited fine-tuning. We also used a full-image fallback for 82 implausible masks; that is an automatic flag rate, not measured segmentation accuracy.”

## Slide 8 — Live demonstration and this term’s work

“This term we implemented original-pixel lung cropping, added a small transformer, ran four controlled comparisons and verified the results. We also packaged a standalone web app, portable weights, report and GitHub presenter hub. To demonstrate, select an image and click Analyze. Show the full image, mask, crop and result. Built-in masks are prepared, but classification runs now; uploads run segmentation too. If setup fails, show the clearly labelled recorded backup.”

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
