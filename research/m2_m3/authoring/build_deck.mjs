import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='/Users/zebra/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const root='/Users/zebra/cvproj/m2_m3';
const skill='/Users/zebra/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const py='/Users/zebra/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python';
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont({fontFamily:'Arial'});
let result=JSON.parse(await fs.readFile(root+'/results/summary.json','utf8'));
const initialMetrics=result.majority_vote;
let fresh=false;
try {
 const receipt=JSON.parse(await fs.readFile(root+'/experiments/restart15/results/independent_verification.json','utf8'));
 if(receipt.verified){const restart=JSON.parse(await fs.readFile(root+'/experiments/restart15/results/summary.json','utf8'));result={...result,majority_vote:restart.metrics};fresh=true;}
}catch{}
let online=null;
try {const receipt=JSON.parse(await fs.readFile(root+'/experiments/paper_online5/results/independent_verification.json','utf8'));if(receipt.verified)online=JSON.parse(await fs.readFile(root+'/experiments/paper_online5/results/summary.json','utf8'));}catch{}
if(online)result={...result,majority_vote:online.metrics};
const resultDir=fresh?root+'/experiments/restart15/results':root+'/results';
const audit=JSON.parse(await fs.readFile(root+'/results/dataset_audit.json','utf8'));
const m=result.majority_vote;
const pct=x=>(100*x).toFixed(2)+'%';const dec=x=>x.toFixed(3);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#0F1B24',body:'#2B3D48',accent:'#0D6E80',muted:'#64757F',bg:'#FDFEFF',line:'#C3D0D8'};
function text(s,value,x,y,w,h,size=27,bold=false,color=C.body){
 const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=value;shape.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return shape;
}
function slide(title,notes){
 const s=p.slides.add();s.background.fill=C.bg;
 text(s,title,64,44,1152,84,44,true,C.ink);
 text(s,'Reading the Reading Room   /   Milestones 2 and 3',64,676,1040,22,16,false,C.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1165,676,50,22,16,false,C.muted);
 s.speakerNotes.textFrame.setText(notes);return s;
}
function table(s,values,x,y,width,height,colWidths,size=24){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,values,columnWidths:colWidths});
 t.borders.assign({fill:C.line,width:1,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{typeface:font,fontSize:size,color:C.body},margins:{left:12,right:10,top:8,bottom:8},anchor:'center'});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#DCEBEE',textStyle:{typeface:font,fontSize:size,bold:true,color:C.ink}});
 return t;
}
let s=slide('Reading the Reading Room','Samarth opens. We study pneumonia classification on pediatric X-rays. M2 reconstructs a five-CNN baseline. M3 proposes one input preprocessing change before testing it. The result numbers are from our saved local outputs.');
text(s,'Pneumonia classification from chest X-rays',64,180,1090,86,48,true,C.accent);
text(s,'Three baseline experiments, with declared reconstruction limits\nLung cropping with a CNN and ViT ensemble proposal',64,302,1080,130,31);
text(s,'Samarth Shekhar  230042     Meghavi Rao  230044\nRachit Kumar  230128          Harshal Nerpagar  230076',64,510,1140,90,26);

s=slide('Introduction and project objective','Meghavi. X-rays contain visual patterns such as lung opacity. Transfer learning supplies features from ImageNet. The aim is first to reconstruct a published baseline, then test one planned change. These are benchmark results, with no clinical claim. Sources: https://doi.org/10.17632/rscbjbr9sj.2 ; https://doi.org/10.3390/app10020559');
text(s,'Task',64,166,200,44,30,true,C.accent);
text(s,'Classify each pediatric chest X-ray as normal or pneumonia.',64,221,1120,76,31);
text(s,'M2 objective',64,326,360,44,30,true,C.accent);
text(s,'Reconstruct the published ensemble and report its actual errors.',64,379,1120,78,31);
text(s,'M3 objective',64,494,360,44,30,true,C.accent);
text(s,'Propose lung cropping and a pretrained ViT, with separate controls.',64,549,1120,80,30);

s=slide('Literature reviewed in M1','Rachit summarizes only the lineage, not every result. We now pick P7. P9 is a useful recent comparison but it reports F1 and confidence intervals and uses a different split. Do not repeat the older claim that all papers use the identical test set. Sources for chosen papers: https://doi.org/10.3390/app10020559 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10827725/');
table(s,[['Paper','Main idea'],['P1  Kermany 2018','Pediatric CXR dataset and transfer learning'],['P2  CheXNet 2017','DenseNet for chest X-ray classification'],['P3  Wang 2017','Large chest X-ray dataset and localization benchmarks'],['P4  Stephen 2019','Train a compact CNN from scratch'],['P5  Ayan and Unver 2019','Compare VGG16 and Xception'],['P6  Rahman 2020','Compare pretrained CNN backbones'],['P7  Chouhan 2020','Five CNNs combined by majority vote   SELECTED'],['P8  Rajaraman 2018','CNN classification and activation-map interpretation'],['P9  Singh 2024','Pretrained DeiT-Base Vision Transformer']],64,145,1152,498,[355,797],21);

s=slide('Selected baseline and reconstruction protocol','Samarth. We switch the tentative M1 Kermany baseline to P7, retaining cropping. No author code was verified. This is an independent reconstruction using torchvision. Figures4-8 specify frozen EARLY and trainable LATE blocks/classifiers. AlexNet replaces its classifier with512 hidden units and two outputs. Five local epochs differ from the paper schedule. Exact departures appear in the notebook/report. Source: P7 Figures4-8 and Table1.');
text(s,'P7  Chouhan et al. 2020',64,151,1120,52,34,true,C.accent);
text(s,'AlexNet, DenseNet121, InceptionV3, ResNet18 and GoogLeNet\nFrozen early layers, trained late blocks, three-of-five vote',64,222,1130,105,28);
table(s,[['Paper partition','Normal','Pneumonia','Total'],['Train + original val','1,349','3,883','5,232'],['Original test','234','390','624']],64,346,1152,169,[500,215,235,202],24);
text(s,'The 15-epoch audit held out 1,042 images and trained on 4,164.\nThe corrected five-epoch run uses all 5,232 with fresh image augmentation.\nPaper schedules are 50 to 200 epochs. Author settings remain partly unknown.',64,550,1140,98,22,false,C.muted);

s=slide('Three runs and their measured results','Every speaker uses the same shared guide. Run1: original five epochs,5232 train, two fixed cached views. Run2: fifteen epochs,4164 train1042 group validation, four cached views cycled by epoch, validation-selected weights. Run3: five epochs,5232 train, full FP32 image forwards, fresh augmentation per example each epoch, ordinary BatchNorm running-statistics updates with paper weight boundaries, fixed final checkpoint. Pending has no final score. These bundled changes do not isolate epochs. All test scores use original624 records, fixed0.5 components and majority vote.');
const rows=[['Run','Epochs','Train','Accuracy','Precision','Status'],['Initial cached','5','5,232',pct(initialMetrics.accuracy),pct(initialMetrics.precision),'Complete'],['Cached audit','15','4,164','85.74%','81.55%','Complete'],['Online correction','5','5,232',online?pct(online.metrics.accuracy):'Pending',online?pct(online.metrics.precision):'Pending',online?'Verified':'Training']];
table(s,rows,64,170,1152,280,[310,110,150,190,190,202],24);
text(s,'Run 3 removes feature caching and samples new augmentation each epoch.\nIt restores the paper training partition and uses fixed epoch-five weights.',64,490,1130,100,27);
text(s,'624 test records for every measured row. Crop and ViT remain untested.',64,611,1130,35,23,false,C.muted);

s=slide('Final results compared with the paper','Say: Our ensemble correctly classified 92.15% of the 624 test images. Precision is close to the paper, but accuracy and pneumonia recall remain lower. This is a verified five-epoch baseline, not an exact match to the paper. The paper trained individual models for50-200 epochs and leaves several settings unspecified, so the gap cannot be attributed to epochs alone. Paper source: https://doi.org/10.3390/app10020559 . Our source: experiments/paper_online5/results/summary.json and independent_verification.json.');
table(s,[['Metric','Paper reported','Our final 5 epochs'],['Accuracy', '96.39%',pct(m.accuracy)],['Pneumonia precision','93.28%',pct(m.precision)],['Pneumonia recall','99.62%',pct(m.sensitivity)],['ROC AUC','99.34%',pct(m.auc)]],64,165,1152,310,[492,330,330],27);
text(s,'Precision is close to the paper. Accuracy is 4.24 percentage points lower.',64,515,1152,66,28,true,C.accent);
text(s,'Our test: 624 images. Our training: 5 epochs per model. Paper: 50 to 200.\nSeveral author settings remain unspecified, so this is a partial reproduction.',64,602,1152,62,22,false,C.muted);

s=slide('M3 proposed crop and CNN ViT system','Rachit. The final proposal combines two interventions. Cropping remains the primary handout hypothesis, and the transformer is a separately tested extension. A fixed pretrained segmenter supplies both lung masks, crop union with10% margin on original pixels before224 resize. Start with ImageNet-pretrained DeiT-Tiny, not P9 DeiT-Base. Use probability aggregation with validation-only weights/thresholds and matching CNN-only aggregation controls. Neither crop nor ViT has run. Sources: https://github.com/mlmed/torchxrayvision ; https://github.com/facebookresearch/deit ; existing hybrid https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/');
text(s,'Lung-focused input, complementary classifiers',64,154,1130,64,34,true,C.accent);
text(s,'1  Crop both lungs with a 10% margin\n    More input resolution for lung tissue, less peripheral variation',64,249,1130,112,29);
text(s,'2  Combine the five CNNs with a pretrained small ViT\n    Patch self-attention may contribute different predictions',64,405,1130,112,29);
text(s,'Two changes, separate hypotheses. Crop is the primary M3 change.\nWeights and thresholds use validation only. Both improvements remain untested.',64,574,1130,73,23,false,C.muted);

s=slide('Comparisons and prediction before implementation','Harshal. A/B/C/D are planned, not results. B-A tests cropping, C-A tests ViT, D-B tests ViT on crops, D-C tests cropping in the hybrid. All use matched groups, per-model schedules and probability aggregation. Original P7 majority vote is separate, so improved voting cannot masquerade as ViT benefit. Crop prediction B-A macroF1+.01-.03, specificity+2-5pp. ViT prediction matched control macroF1+.01-.03 and accuracy+1-3pp. Combined D-A macroF1+.01-.04 and accuracy0-3pp with recall loss no more than2pp. Bounds cap at metric maxima and are not additive. JEPA and mixup remain optional future work.');
table(s,[['Run','Image region','Classifiers'],['A','Full X-ray','Five CNNs'],['B','Lung crop','Five CNNs'],['C','Full X-ray','CNNs plus pretrained ViT'],['D  Final proposal','Lung crop','CNNs plus pretrained ViT']],64,155,1152,300,[260,320,572],27);
text(s,'Combined prediction versus A',64,491,1130,44,29,true,C.accent);
text(s,'Macro F1 +0.01 to +0.04. Accuracy 0 to +3 percentage points.\nPneumonia recall loss no more than 2 points. Gains need not add up.',64,553,1130,75,26);

s=slide('Live demo and next experiment','Samarth runs the live demo: choose a held-out test sample, click Process X-ray, explain the five-CNN vote and ResNet18 component Grad-CAM. Then upload a known image. The score is uncalibrated and heatmap is not lesion localization or ensemble-wide attribution. The screenshot is an actual tested live response. M4 compares matched full-image/crop models after submitting M3. No clinical claim. Hybrid models already exist: https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/.');
const demoImage=await fs.readFile(root+'/.build/demo_redesign_desktop.png');
s.images.add({blob:new Uint8Array(demoImage),contentType:'image/png',alt:'Actual tested local X-ray prediction demo',fit:'contain',position:{left:64,top:145,width:740,height:485}});
text(s,'Real inference',841,169,370,45,31,true,C.accent);
text(s,'Test samples or upload\nFive trained CNN votes\nResNet18 Grad-CAM',841,235,372,135,27);
text(s,'Next: crop plus ViT',841,407,372,46,30,true,C.accent);
text(s,'Four matched comparisons\nNo new crop or ViT results\nJEPA and mixup later',841,470,372,123,25);
text(s,'Local demo: 127.0.0.1:8765. Demo uses original CNN weights. No clinical claim.',64,635,1140,33,21,false,C.muted);

p.slides.items[0].speakerNotes.textFrame.setText('All four members receive the same preparation guide, with no assigned slides. M2 reconstructs P7 through three declared experiments. M3 proposes lung cropping plus a pretrained small ViT with separate controls. Neither improvement has run. The shared guide and slide5 distinguish completed and pending scores.');
p.slides.items[1].speakerNotes.textFrame.setText('Meghavi. The task is normal-versus-pneumonia pediatric X-ray classification. Transfer learning uses features learned from larger datasets. M2 establishes the runnable baseline. M3 proposes cropping plus a ViT with separate controls. These are student benchmark experiments, with no clinical claim.');
p.slides.items[3].speakerNotes.textFrame.setText('P7 Figures4-8 specify early frozen weights and trainable late blocks/classifiers. Table1 has5232 training and624 test records. Run1 follows those counts with fixed cached features. Run2 deduplicates/reserves1042 validation and trains4164; selected epochs13,9,5,13,1 follow the predeclared validation criterion. Run3 restores5232 records, five fixed epochs and full-image FP32 training with fresh augmentation; BatchNorm running statistics update but early weights stay frozen. The paper schedules50-200 epochs and leaves author settings unspecified. The benchmark was inspected previously, not a blind holdout.');
p.slides.items[8].speakerNotes.textFrame.setText('Samarth demonstrates the existing verified five-epoch CNN demo, which remains unchanged. Choose a sample or upload a PNG/JPEG, process, then explain the ResNet18 component Grad-CAM. The demo has no cropped input or ViT yet and is not a clinical diagnosis. Our proposed next system includes both, evaluated using A/B/C/D controls. JEPA and mixup are optional later experiments, not submission results.');

await fs.mkdir(root+'/.build/rendered',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(root+'/.build/rendered/slide-'+(i+1)+'.png',new Uint8Array(await blob.arrayBuffer()));
}
const candidate=root+'/.build/Reading_Room_candidate.pptx';
await (await PresentationFile.exportPptx(p)).save(candidate);
const revision=process.env.DECK_REVISION||'shared_v6';
const final=root+'/.build/share/Reading_Room_'+revision+'.pptx';
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--cover-role','cover',...[3,4,5,6,8].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:9,requiredNativeTableOwnerSlides:[3,4,5,6,8],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:root+'/.build/deck_validation_'+revision+'.json'});
await fs.copyFile(final,root+'/output/Reading_Room_M2_M3.pptx');
console.log(final);
