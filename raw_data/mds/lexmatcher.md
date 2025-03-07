# LexMatcher: Dictionary-centric Data Collection for LLM-based Machine Translation  

Yongjing $\mathbf{Y}\mathbf{in}^{1,2}$ , Jiali $\mathbf{Zeng^{4}}$ ,Yafu $\mathbf{Li^{2}}$ , Fandong Meng4, Yue Zhang2.3\* 1Zhejiang University 2School of Engineering, Westlake University   
3Institute of Advanced Technology, Westlake Institute for Advanced Study 4Pattern Recognition Center, WeChat AI, Tencent Inc {yinyongjing,liyafu}@westlake.edu.cn {lemonzeng, fandongmeng}@tencent.com yue.zhang@wias.org.cn  

## Abstract  

The fine-tuning of open-source large language models (LLMs) for machine translation has recently received considerable attention, marking a shift towards data-centric research from traditional neural machine translation. However, the area of data collection for instruction finetuning in machine translation remains relatively underexplored. In this paper, we present LexMatcher, a simple yet effective method for data collection that leverages bilingual dictionaries to generate a dataset, the design of which is driven by the coverage of senses found in these dictionaries. The dataset comprises a subset retrieved from an existing corpus and a smaller synthesized subset which supplements the infrequent senses of polysemous words. Utilizing LLaMA2 as our base model, our approach outperforms the established baselines on the WMT2022 test sets and also exhibits significant performance improvements in tasks related to word sense disambiguation and specialized terminology translation. These results underscore the effectiveness of LexMatcher in enhancing LLM-based machine translation.  

## 1 Introduction  

The emergence of large language models (LLMs) has brought about new opportunities for machine translation. The modeling paradigm has gradually shifted from training sequence-to-sequence models from scratch to utilizing commercial or opensourced LLMs (Hendy et al., 2023; Agrawal et al., 2023; Zhu et al., 2023; Jia0 et al., 2023; Xu et al., 2024). Unlike traditional neural machine translation methods, which rely on abundant parallel sentences (Vaswani et al., 2017; Gordon et al., 2021; Fernandes et al., 2023) and monolingual sentences (Sennrich et al., 2016; Edunov et al., 2018), it has been shown that LLMs do not require much supervised fine-tuning data to achieve competitive translation, and the quality of fine-tuning data is crucial (Zhang et al., 2023; Xu et al., 2024).  

Current research primarily focuses on constructing fine-tuning data by leveraging human-written development sets, as well as creating refined instruction data to improve performance, such as contrastive translation pairs and interactive translation (Jiao et al., 2023; Zeng et al., 2024; Zhang et al., 2023). In addition, some studies have explored post-training LLMs using extensive bilingual data (Yang et al., 2023; Wei et al., 2023). However, what kind of fine-tuning data is important for machine translation has not been thoroughly investigated yet. It has been demonstrated that fine-tuning LLMs with extensive parallel data can harm their intrinsic translation capabilities (Xu et al., 2024). Furthermore, recent studies emphasize that the quality of data distributions has a more significant impact on pretraining outcomes than quantity alone (Gunasekar et al., 2023; Li et al., 2023), with more uniform data distributions contributing to improved generalization for unseen compositions (Patel et al., 2022).  

Motivated by the above observations, we investigate a principled method, LexMatcher, for curating supervised fine-tuning data for LLM-based translation. The objective is to collect a small yet carefully selected dataset that follows a proper distribution for maximizing translation quality. In particular, we leverage a bilingual dictionary as a pivotal resource to ensure comprehensive coverage of word or phrase senses in bilingual contexts. The construction of the dataset involves two steps: sense retrieval and sense supplement. In the sense retrieval step, we traverse commonly-used corpora (e.g., WMT training data), and identify sentence pairs that contain at least a segment pair that has not yet reached a matching threshold in the retrieved subset. To prioritize the selection of high-quality sentence pairs, we employ a quality estimation model to sort the data. Inevitably, there may be uncovered senses of polysemous words after the retrieval, representing crucial long-tail knowledge essential for accurate translation. To address this, we employ commercial LLMs (e.g., ChatGPT) to generate precise and concise demonstrations for the uncovered senses. Finally, we fine-tune LLMs using a combination of the retrieved and synthesized subsets.  

We primarily utilize the training data from WMT22 and conduct extensive experiments on six language directions, including $\mathrm{Zh}{\Leftrightarrow}\mathrm{En}$ ， $\mathrm{En{\Longleftrightarrow}D e}$ ， and $\mathtt{E n}{\Longleftrightarrow}\mathtt{R u}$ . By employing LexMatcher, we extract $0.1\%$ of the data from the original corpus and utilize a maximum of only 1 million samples across all six language directions. The results of fine-tuned LLMs on the WMT22 test sets deliver superiority over the baselines in both common and zero-shot settings. The fine-tuned models also achieve comparable or better performance in terminology translation and translation disambiguation compared to the dedicated or commercial systems. Furthermore, the analyses of different data collection methods and composition generalization underscore the significance of high-quality data distributions. Finally, we showcase the complementarity of the collected parallel data with large-scale monolingual post-training by the experiment of fine-tuning ALMA (Xu et al., 2024). The code, data, and models are available at https://github.com/ARIESLM/Lexmatcher-MT.git.  

## 2 Related Work  

Data Selection for NMT. For traditional sequence-to-sequence neural machine translation (NMT) models, augmenting the volume of parallel data often leads to improvements in performance (Sennrich et al., 2016; Edunov et al., 2018; Gordon et al., 2021; Fernandes et al., 2023). Conversely, there have also been studies exploring data selection to reduce the size of the training corpus. For instance, van der Wees et al. (2017) gradually reduces the training data to a cleaner subset, determined by external scorers. Wang et al. (2018) introduce curriculum-based data selection that employs a trusted clean dataset to assess the noise level of each sample. Kumar et al. (2019) employ reinforcement learning to simultaneously learn a denoising curriculum and improve the NMT model. Mohiuddin et al. (2022) initially train a base NMT model on the entire available data and subsequently fine-tune the base model using selected subsets of the data. Compared to traditional NMT, data collection is more critical for LLM-based MT, yet it remains unexplored. We have taken the initiative to investigate it and propose a simple and practical method.  

LLMs for MT. The usage of LLM-based MT is significantly different from the conventional NMT. LLMs, particularly large ones like GPT-4, serve as interfaces that can perform translation with simple translation instructions or ICL (Lin et al., 2022; Hendy et al., 2023; Zhu et al., 2023; Agrawal et al., 2022). For ICL, the influence of data selection methods on model performance is not significantly noticeable (Zhu et al., 2023; Agrawal et al., 2022; Lin et al., 2022). Fine-tuning open-source LLMs such as LLaMA (Touvron et al., 2023) for translation has garnered increasing attention (Jiao et al., 2023; Zhang et al., 2023). TIM (Zeng et al., 2024) constructs translation pairs for comparison and introduces an additional preference loss during SFT. Bayling (Zhang et al., 2023) automatically generates interactive translation instructions for tuning. (Mao and Yu, 2024) construct an additional cross-lingual discrimination task using word alignment for instruction fine-tuning in low-resource languages. Yang et al. (2023) fine-tune LLMs using more than 300 million parallel instances while Xu et al. (2024) indicate that such strategy could potentially impair the translation capabilities of LLMs. Instead, they propose a two-stage process that involves further post-training LLMs using a substantial amount of mixed monolingual data, followed by a subsequent step of SFT with humanwritten parallel data.  

In contrast, we are the first to propose specific parallel data collection methods, following the principle of achieving uniform coverage of semantic units in the dictionary. Moreover, our approach achieves a better balance between efficiency and performance, i.e., a high-quality translation model can be obtained using less computational resources than monolingual or bilingual post-training.  

Bilingual Dictionary for NMT. Bilingual dictionaries have been employed to enhance translation quality, particularly for rare words or domainspecific entities. One approach involves augmenting the training data with pseudo-parallel sentences generated based on the dictionary. For example, Zhao et al. (2020) enhance the parallel corpus with the help of paired entities extracted from multilingual knowledge graphs. Hu et al. (2022) propose denoising entity pretraining for NMT using monolingual data and paired entities. These methods do not consult bilingual dictionaries for translation candidates during the inference stage. Another approach involves leveraging bilingual alignments as lexical constraints (Li et al., 2022; Wang et al., 2022; Zeng et al., 2023). For LLMs, bilingual dictionaries have been used as a part of prompts (Lu et al., 2023; Ghazvininejad et al., 2023) for the LLMs more than 100B. In comparison, we aim to improve LLMs’ fine-tuning performance on translation tasks. The dictionaries serve as a pivot for data collection and can also be added in prompts whenneeded.  

![](https://cdn-mineru.openxlab.org.cn/extract/e7af52ed-8b9f-4b8f-92be-1462377e55cd/1d7a9cbca50515f163eb3197a5b894fa129e2e69c0c5c76ccd8efbf9b7872671.jpg)  
Figure 1: Illustration of our LexMatcher for instruction fine-tuning.  

## 3 Method  

The overview of LexMatcher is illustrated in Figure 1. In brief, LexMatcher can generate a compact parallel dataset for instruction fine-tuning based on the provided dictionaries and corpus.  

### 3.1 Sense Retrieval  

Given  a  dictionary $\begin{array}{r c l}{\Phi}&{=}&{(s,t)}\end{array}$ ， where $\Phi=$ $\left\{{{\left({{s_{1}},{t_{1}}}\right)},{\left({{s_{2}},{t_{2}}}\right)},\ldots,{\left({{s_{n}},{t_{n}}}\right)}}\right\}$ and each $(s_{i},t_{i})$ represents a source-target segment pair, we aim to ground each pair in parallel contexts by retrieving data from a parallel dataset $D=(x,y)$ . The dictionary $\Phi$ shares the same source and target languages with $D$ . In certain cases, the segments can be phrases (e.g., “take over') or named entities (e.g., "World Trade Organization") in the dictionary. Ideally, the objective is to find a subset $S_{r}\subseteq D$ such that:  

$$
\forall(s,t)\in\Phi,\exists(x,y)\in S_{r}:s\subseteq x\land t\subseteq y,
$$  

where $\begin{array}{r l r}{x}&{{}=}&{\{x_{1},x_{2},...,x_{|x|}\}}\end{array}$ and $y\quad=$ $\{y_{1},y_{2},...,y_{|y|}\}$ . In practice, we cannot guarantee that the existing bilingual corpora can cover all senses in the dictionary, and we extract a subset that satisfies this objective to the full.  

We traverse the corpus in sequential order and search for potential matches with paired words in the dictionary. To prioritize the extraction of high-quality sentence pairs, we rank the corpus with model-based translation quality metrics, e.g., COMET-KIWI (Rei et al., 2022). Specifically, for each segmentl in a source sentence, we perform a dictionary lookup for all the aligned target words. If one of the aligned target segments exists in the target sentence, we put the sentence pair into the translation candidate subset $S_{r}$ We lemmatize each word in the source and target sentence to alleviate the effect of textual variations. In addition, we introduce a threshold $K$ to skip the sentence if all the segment pairs in it have already been matched $K$ times. $K$ enables convenient control over the size of the subset and is used to encourage the even distribution of segment pairs to some extent. The matching procedure is illustrated in Algorithm 1.  

### 3.2 Sense Supplement  

Using a partial set of open-source corpora cannot cover all the senses in the dictionary, and some senses may be included in the filtered low-quality data. The unseen senses could be named entities or simply low-frequency occurrences. The translation of rare entities is generally unique and can be solved effectively by prompting LLMs during inference, and the lack of training data for these cases may have minimal impact. In contrast, the senses of polysemous words are context-sensitive and may require specific training data to strengthen the model's understanding and translation of these words. To compensate for the missing senses, we leverage ChatGPT2 to construct translation demonstrations for each sense, thus creating the subset $S_{c}$ . Concretely, we prompt ChatGPT with a sense expressed in source and target languages and the sense's definition. The prompt is shown in Figure 6 (Appendix B). Only nouns and verbs  

# Algorithm 1 Sense Retrieval in LexMatcher  

1:Input:Parallel dataset $D$ , dictionary $\Phi$ , thresh  
old $K$   
2:Output:Subset $S_{r}\subseteq D$   
3: Initialize $S_{r}=\emptyset$ ,frequency count $C=\{\}$   
4: for each $(x,y)\in\bar{D}$ do   
5: InitializeFound=false   
6: for each segment $\hat{x_{i}}$ in Lemmatize $(x)$ do   
7: for each $t_{n}$ in $\Phi[\hat{x_{i}}]$ do   
8: if $C[(\hat{x_{i}},t_{n})]~<~K$ and $t_{n}$ in   
Lemmatize $(y)$ then   
9:   
10: Set Found=true   
11: end if   
12: end for   
13: end for   
14: if Found then   
15: Add $(x,y)$ to $S_{r}$   
16: end if   
17: end for   
18:return $S_{r}$  

with more than three senses are considered due to their highly polysemous nature (Campolungo et al., 2022). Note that the subset $S_{c}$ onlytakes up a neglectable portion of the whole dataset, e.g.. 225 sentence pairs for English-Germen, and the specific numbers are reported in the experiment.  

Table 1: The number of parallel sentences of different data sets.   


<html><body><table><tr><td rowspan="2">Lang</td><td rowspan="2">Raw</td><td colspan="3">Retrieval</td><td rowspan="2">Supplement</td></tr><tr><td>K=1</td><td>K=2</td><td>K=3</td></tr><tr><td>Zh</td><td>33M</td><td>75k</td><td>188k</td><td>281k</td><td>2.2k</td></tr><tr><td>De</td><td>278M</td><td>93k</td><td>233k</td><td>351k</td><td>0.2k</td></tr><tr><td>Ru</td><td>227M</td><td>98k</td><td>246k</td><td>367k</td><td>0.7k</td></tr></table></body></html>  

dictionary or based on specific user requirements. To train the LLM in both scenarios - with and without specified word translations - we adopt a sampling strategy inspired by Li et al. (2022). We randomly select a small number of sentence pairs to incorporate specified word translations?. For each chosen instance, we sample at most 3 segment pairs that are matched in the dictionary and construct corresponding instructions with a template:  

$$
c=\mathrm{Template}(\{(s_{i},t_{i})\}_{i=1}^{N}).
$$  

For the template selection, we simply use “means" to connect $s_{i}$ and $t_{i}$ , and prepend the constraint to the instruction. An example is shown in Figure 6 (Appendix B). In this way, we can choose whether to incorporate translations from the dictionary as auxiliary information during inference, depending on the situation.  

## 4 Experiments  

### 3.3  Instruction Fine-tuning  

Instruction fine-tuning has become standard practice in LLM-based translation (Jiao et al., 2023; Xu et al., 2024; Zhang et al., 2023). The instructionfollowing data is constructed based on $S=S_{r}\cup S_{c}$ Generally, each instance comprises an “instruction" $c$ describing the task the model should perform (e.g., “Translate the sentences from English to Chinese."), an “input" $x$ indicating the source sentence, and a corresponding output $y$ indicating the answer to the instruction, i.e., the target sentence. The language models are optimized by minimizing the negative log-likelihood of the output $y$  

$$
L=-\sum_{(x,y)\in S}\frac{1}{|y|}\sum_{i}^{|y|}\log p(y_{i}|c,x;\theta),
$$  

where $\theta$ is the trainable parameters. We use two kinds of translation instructions: 1) general translation instructions mainly used to indicate translation directions, and 2) constrained translation instructions that specify word translations from a given  

### 4.1  Setting  

For parallel training data, we use the opensource data from WMT22 in German $\Leftrightarrow$ English, Chinese $\Leftrightarrow$ English, and Russian $\Leftrightarrow$ English. The detail of data preprocessing is shown in Appendix C. We use bilingual dictionaries provided by Open Multilingual WordNet (Bond et al., $2016)^{4}$ .In addition, we involve Wikititles? as an entity dictionary. Table 1 presents the number of sentence pairs for each language pair in different subsets, including the original training set, subsets extracted based on different $K$ , and the ChatGPT-generated data. It can be observed that our method achieves a high compression rate. The subset $K{=}3$ is used for the main experiment, and the extracted data for Chinese, German, and Russian accounts for only $0.57\%$ ， $0.08\%$ , and $0.11\%$ of the original data, respectively. The development sets from the previous  

Table 2: Evaluation results on WMT22 test sets. Higher scores (BLEU and COMET) denote better translation performance. Bold numbers indicate the best scores among models of the same sizes. The numbers with the dagger symbol represent the results from (Xu et al., 2024). LexMatcher-7B outperforms Parrot-7B and ALMA-7B with p-value ${<}0.01$ , and LexMatcher-13B outperforms ALMA-13B with p-value ${<}0.01$   


<html><body><table><tr><td>Model</td><td>Zh=En BLEU/COMET</td><td>En=Zh BLEU/COMET</td><td>De=En BLEU/COMET</td><td>En=De BLEU/COMET</td><td>Ru=En BLEU/COMET</td><td>En=Ru BLEU/COMET</td></tr><tr><td>GPT-3.5t</td><td>26.60/82.90</td><td>44.90/87.00</td><td>33.10/85.50</td><td>34.40/87.00</td><td>42.40/86.10</td><td>34.40/87.00</td></tr><tr><td>GPT-4t</td><td>27.20/82.79</td><td>43.98/87.49</td><td>33.87/85.62</td><td>35.38/87.44</td><td>43.51/86.18</td><td>30.45/88.87</td></tr><tr><td>NLLB-54Bt</td><td>16.56/70.70</td><td>27.38/78.91</td><td>26.89/78.94</td><td>34.50/86.45</td><td>26.89/78.94</td><td>30.96/87.92</td></tr><tr><td>LLaMA2-7Bt</td><td>18.19/75.00</td><td>16.97/71.80</td><td>30.42/82.74</td><td>19.00/76.39</td><td>36.02/82.84</td><td>16.00/73.24</td></tr><tr><td>Parrot-7B (Jiao etal.,2023)</td><td>20.20/75.90</td><td>30.30/80.30</td><td>27.30/82.40</td><td>26.10/81.60</td><td></td><td></td></tr><tr><td>TIM-7B (Zeng et al.,2024)</td><td>24.51/79.71</td><td>37.83/85.10</td><td>26.12/78.94</td><td>20.90/74.91</td><td></td><td></td></tr><tr><td>ALMA-7B (Xuetal.,2024)</td><td>23.52/79.73</td><td>36.48/85.05</td><td>29.49/83.98</td><td>30.31/85.59</td><td>38.93/84.81</td><td>27.09/87.17</td></tr><tr><td>LexMatcher-7B</td><td>24.81/79.13</td><td>40.34/86.11</td><td>32.33/84.29</td><td>33.56/86.31</td><td>41.01/84.43</td><td>28.97/87.23</td></tr><tr><td>LLaMA2-13Bt</td><td>21.81/78.10</td><td>30.00/79.70</td><td>31.06/83.01</td><td>13.69/75.55</td><td>36.50/82.91</td><td>0.59/63.84</td></tr><tr><td>DictPrompt-13B(Ghazvininejadet al.,2023)</td><td>17.55/74.12</td><td>33.75/83.46</td><td>30.36/83.31</td><td>25.24/80.89</td><td>37.70/81.95</td><td>21.98/81.00</td></tr><tr><td>BigTrans-13B(Yang et al.,2023)</td><td>14.16/74.26</td><td>28.56/81.31</td><td>23.35/80.68</td><td>21.48/78.81</td><td>26.81/77.80</td><td>17.66/78.21</td></tr><tr><td>Bayling-13B(Zhang et al.,2023)</td><td>20.12/77.72</td><td>37.92/84.62</td><td>27.34/83.02</td><td>25.62/82.69</td><td>33.95/82.07</td><td>12.77/71.01</td></tr><tr><td>ALMA-13B (Xuet al.,2024)</td><td>25.46/80.21</td><td>39.84/85.96</td><td>31.14/84.56</td><td>31.47/85.62</td><td>40.27/85.27</td><td>28.96/87.53</td></tr><tr><td>LexMatcher-13B</td><td>26.15/79.88</td><td>41.13/86.58</td><td>32.59/84.55</td><td>34.82/86.45</td><td>41.53/84.91</td><td>30.20/87.83</td></tr></table></body></html>  

WMT competitions are used by default (Jiao et al., 2023; Xu et al., 2024).  

We fine-tune LLaMA2-7B and LLaMA2-13B for 1 epoch with the collected multilingual instruction data. The batch size is 128 and the learning rate is 2e-5. The final checkpoint is used for evaluation, and we use beam search with a beam size of 4 during inference. For automatic evaluations, we use BLEU (Papineni et al., 2002) ° and COMET'.  

### 4.2 Main Results  

Seen Language Directions. Table 2 presents the translation performance on the WMT22 test sets. The LLaMA2 models fine-tuned on the instruction data collected by LexMatcher significantly outperform their original zero-shot performance, especially for the $\mathrm{En}{\Rightarrow}\mathbf{X}\mathbf{X}$ .Concretely, LexMatcher-7B improves LLaMA2-7B by an average of 17.02 BLEU points and 12.68 COMET points in $\mathrm{En}{\Rightarrow}\mathbf{X}\mathbf{X}$ , and by 4.45 BLEU points and 2.42 COMET points in $\scriptstyle\mathbf{X}\mathbf{X}{\stackrel{\mathrm{~\tiny~\left.~\right.~}}{\Rightarrow\mathrm{En}}}$ . LLaMA2-13B performs significantly worse than its 7B counterpart in $\mathrm{En}{\Rightarrow}\mathbf{X}\mathbf{X}$ directions due to severe off-target issues, while LexMatcher-13B improves this performance significantly. We also consider an ICL method DictPrompt (Ghazvininejad et al., 2023) which provides dictionary translations for each source word? and the result shows that using dictionary translations as hints yields notable improvements in $\mathrm{En}{\Rightarrow}\mathbf{X}\mathbf{X}$ .In contrast, LexMatcher-13B achieves better performance and is more efficient due to a much shorter context during inference.  

![](https://cdn-mineru.openxlab.org.cn/extract/e7af52ed-8b9f-4b8f-92be-1462377e55cd/d06181c6174a8163b99eceb1a4e88b0613b923c6586352cb27c903ccd3be66dd.jpg)  
Figure 2: Zero-shot translation.  

LexMatcher demonstrates superior performance compared to other instruction fine-tuned baselines. Specifically, LexMatcher-7B outperforms Parrot-7B and TIM-7B, which construct additional translation pairs and utilize specialized instructions. In the $\mathrm{En{=}D e}$ translation task, LexMatcher7B surpasses TIM-7B by more than 10 BLEU and COMET points.  Moreover, LexMatcher outperforms BigTrans and ALMA consistently across the $\mathrm{En}{\Rightarrow}\mathbf{X}\mathbf{X}$ tasks, which incorporate a large amount of data for continual pretraining. While LexMatcher-7B still underperforms GPT- $3.5^{9}$ and GPT- $4^{10}$ , the COMET scores for LexMatcher-7B are merely lower than GPT-3.5 within 2 points, and LexMatcher-13B further narrows the gap.  

Table 3: Accuracies on the DiBiMT benchmark which is dedicated for evaluating word disambiguation in MT. The number following ICL denotes the number of translation demonstrations.   


<html><body><table><tr><td>Model</td><td>Zh</td><td>De</td><td>Ru</td></tr><tr><td>DeepL Google-Translate OPUS NLLB-54B</td><td>58.42 52.09 25.94 48.02</td><td>76.64 67.35 27.04 67.97</td><td>67.53 62.03 28.71 67.88</td></tr><tr><td>LLaMA-7B-ICL(1) LLaMA-7B-ICL(5) LLaMA-65B-ICL(1) LLaMA-65B-ICL(5) Alpaca-7B</td><td>30.61 27.92 44.73 42.49 29.63</td><td>57.41 55.26 62.05 62.98 51.52</td><td>60.65 56.83 65.71 66.31 55.23</td></tr><tr><td>LexMatcher-7B LexMatcher-13B</td><td>53.28 59.09</td><td>63.32 66.98</td><td>67.72 69.93</td></tr></table></body></html>  

Unseen Language Directions. To evaluate performance in translation directions never seen previously, i.e., zero-shot multilingual capability, we further conduct experiments on Czechto-English $(\mathbf{cs}\Rightarrow\mathbf{en}$ 0, Japanese-to-English $(\mathrm{ja}{\Rightarrow}\mathrm{en})$ ， and Ukrainian-to-English $(\mathbf{u}\mathbf{k}{\Rightarrow}\mathbf{en}$ ). As depicted in Figure 2, LexMatcher- $({}^{*})$ exhibits superior zeroshot multilingual capability over the LLM baselines, highlighting that better aligning training languages strengthens the alignment of other languages as a by-product.  

Disambiguation.  By comparing the different senses of a word and multilingual expressions of meaning, the model possibly learns more precise word usage in translation. To investigate it, we submit the models to a challenging disambiguation leaderboard, DiBiMT (Campolung0 et al., 2022). It compares the performance of NMT systems when translating sentences with ambiguous words and the performance is evaluated by accuracy. For comparison, we display the performance of top-ranked systems including DeepL11, Google Translate12, and NLLB-54B. The results of LLMs are from Iyer et al. (2023).  

The result is shown in Table 3. For the LLaMA models, increasing model size improves the performance, and LLaMA-65B matches Google Tranlate and NLLB-54B with few-shot prompting. Alpaca7B works well without demonstration (i.e., zeroshot prompting) and significantly outperforms the supervised NMT system OPUS, which indicates its potential for further improvement through finetuning on translation data. LexMatcher-7B significantly outperforms Alpaca- $^ Ḋ 7B Ḍ$ and surpasses Google Translate in Chinese and Russian disambiguation. With a scale of 13B, it also outperforms the best DEEPL system in Chinese and Russian, achieving accuracy rates of $59.09\%$ and $69.93\%$ ,respectively. This result demonstrates the advantage of our data construction principle.  

Table 4: Performance on WMT23 terminology translation test sets. “Suc" indicates Terminology Success Rate.   


<html><body><table><tr><td>Model</td><td>Zh=En ChrF/COMET Suc</td><td>ChrF/COMET</td><td>De=En Suc</td></tr><tr><td rowspan="3">Lingua Custodia VARCO</td><td></td><td></td><td></td></tr><tr><td>32.6/60.9</td><td>74.7</td><td>61.8/73.5</td></tr><tr><td>40.5/71.5 41.2/75.7</td><td>80.0 75.3</td><td>60.0/81.3</td></tr><tr><td rowspan="3">LexMatcher-7B</td><td></td><td>84.5</td><td></td></tr><tr><td>38.2/73.2</td><td></td><td>64.3/81.9</td></tr><tr><td>39.1/73.6</td><td>85.6</td><td>64.5/82.0 81.5</td></tr></table></body></html>  

Terminology. During training, we introduce special instructions to train the model to use the provided segment pairs. In this experiment, we evaluate the effectiveness of the instructions on a terminology translation test set from $\mathbf{WMT}23^{13}$ . The numbers of sentences on $Z\mathrm{h}{\Rightarrow}\mathrm{En}$ and $\mathrm{De}{\Rightarrow}.$ En are 2640 and 2963, respectively. The average numbers of terms per segment on $Z\mathrm{h}{\Rightarrow}\mathrm{En}$ and $\mathrm{De}{\Rightarrow}\mathrm{En}$ are 3.8 and 1.1, respectively. The result is shown in Table 4, and we only present the systems achieving the best performance on a specific metric (Semenov et al., 2023). Lingua Custodia and VARCO are specialized Transformer architectures to ensure the appearance of given terminology in the translation, and $U E D I N_{\mathrm{LLM}}$ uses ChatGPT with terminology translation prompts. Compared to them, our models achieve significantly higher terminology success rates, indicating a superior ability to accurately respond to the given domain-specific terminology. On the quality metrics, our models are inferior to $U E D I N_{\mathrm{LLM}}$ on $Z\mathrm{h}{\Rightarrow}\mathrm{En}$ , and achieve the best results on $\mathrm{De}{\Rightarrow}\mathrm{En}$  

## 5 Analysis  

### 5.1 Effect of $K$  

The maximal number of bilingual contexts of each matched sense is influenced by $K$ .We show the performance of varying $K\mathbf{s}$ across different model sizes on the WMT22 test sets (Figure 3). Regardless of the amount of training data used, the larger models perform better and require less data for fine-tuning. In addition, the model's performance improves as $K$ increases from 1 to 3. With the addition of more parallel data, the performance gains begin to plateau or even slightly decrease, which aligns with the conclusions of the previous study $\mathrm{{{Xu}}}$ et al., 2024). Thanks to the strong fewshot learning capability of the backbones, we do not need to provide as many training examples as before when training the NMT model.  

![](https://cdn-mineru.openxlab.org.cn/extract/e7af52ed-8b9f-4b8f-92be-1462377e55cd/dae206d1c71f0ac52373b6ce350867898db78a47659d5a3854a68094d14614b1.jpg)  
Figure 3: BLEU and COMET on the WMT22 test sets with varying $K$ and model sizes.  

### 5.2   Effect of Selection Strategies  

In this experiment, we investigate two intuitive data collection methods: 1) random selection (RAND), in which the training data are randomly sampled from the corpus; and 2) quality-based selection $(T O P)$ , in which the training samples are selected based on the COMET-KIWI scores in descending order. Specifically, we use these two methods to extract the same sample quantity as LexMatcher to mitigate the impact of sample quantity. We use LLaMA2-7B as the backbone, and the result on WMT test sets is shown in Figure 4. The performance of RAND is inferior to the other two methods. Random selection ensures a certain degree of diversity but the performance is uncontrollable and non-reproducible. TOP performs better than RAND, demonstrating the importance of data quality for instruction tuning. LexMatcher can simultaneously consider both quality and diversity and achieve the best performance.  

![](https://cdn-mineru.openxlab.org.cn/extract/e7af52ed-8b9f-4b8f-92be-1462377e55cd/9d84b2215b47e9fc04d3c6167deacf2b4c9b3ba4d0ff43836c768b61d3ea1088.jpg)  
Figure 4: Performance of different data selection strategies.  

![](https://cdn-mineru.openxlab.org.cn/extract/e7af52ed-8b9f-4b8f-92be-1462377e55cd/f50eced64e1cc22217b3fdec246353c379cac748b67b43eeed779e070ba7be71.jpg)  
Figure 5: Word frequency distributions. The blue and gray curves denote the distributions calculated on the data selected by LexMatcher $(\mathrm{K}{=}1)$ ) and randomly selected data, respectively.  

Word Frequency Distribution   We are interested in whether the collected data has a different word frequency distribution from the general (randomly selected) one. We use the English data of the $\mathrm{EN}{\Rightarrow}Z\mathrm{H}$ translation task with $K{=}1$ , and plot the word frequency distributions of the collected data (blue curve) and the corresponding random data (gray curve). As shown in Figure 5, the blue curve tends to be smoother than the gray one, and the blue curve has more flat segments. For words with higher frequency rankings, the word frequency of the data selected based on the dictionary is lower than that of the random data. This phenomenon indicates that the dictionary-based method has generated a less skewed data distribution, which could be the reason for better fine-tuning performance. Additionally, the dictionary-based data contains $98\mathrm{k}$ unique words while the random data only includes $62\mathrm{k}$ unique words, indicating that the dictionarybased data covers more semantic units, thus diluting the word frequency.  

Table 5: Ablation study on different data subsets.   


<html><body><table><tr><td>Model</td><td>xx=En BLEU/COMETBLEU/COMET</td><td>Enxx</td><td>DiBi-Acc</td></tr><tr><td>Dev</td><td>29.77/82.05</td><td>29.41/84.63</td><td>55.51</td></tr><tr><td>+Supplement</td><td>30.39/82.22</td><td>30.10/84.55</td><td>55.96</td></tr><tr><td>+Retrieval</td><td>32.86/82.71</td><td>34.13/86.27</td><td>59.98</td></tr><tr><td>LexMatcher(3)</td><td>32.71/82.61</td><td>34.29/86.55</td><td>61.44</td></tr></table></body></html>  

### 5.3Ablation Study  

The ablation experiment of different data subsets is presented in Table 5. We use LLaMA2-7B as the backbone. Based on the development data, simply incorporating the small amount of synthesized data generated during the sense supplement phase does not have a significant impact on the performance. This is possibly because the data is predominantly focused on low-frequency senses, and the model is unable to effectively leverage this knowledge. In comparison, adding the retrieved data leads to a significant performance improvement, and further introducing the synthesized data helps the model learn word disambiguation better, increasing the disambiguation accuracy from 59.98 to 61.44.  

5.4 Combination with ALMA   
Table 6: Combination with ALMA-7B.   


<html><body><table><tr><td>Model</td><td colspan="2">xx=En XX←UT BLEU/COMETBLEU/COMET</td></tr><tr><td>ALMA</td><td>30.64/82.84</td><td>31.29/85.93</td></tr><tr><td>+LexMatcher(1)</td><td>32.34/83.11</td><td>33.50/86.42</td></tr><tr><td>+LexMatcher(2)</td><td>31.88/83.07</td><td>33.31/86.47</td></tr></table></body></html>  

ALMA (Xu et al., 2024) is the post-trained LLaMA2 on a large amount of monolingual data mixed by different languages. We use ALMA-7B as the backbone and investigate whether the two methods can complement each other. As shown in Table 6, adding the parallel sentences constructed by LexMatcher further improves the performance of ALMA, indicating the compatibility of monolingual continual pretraining and bilingual supervised fine-tuning. Specifically, utilizing the training data With $K{=}1$ is sufficient to enhance ALMA's performance. Our findings indicate that the use of monolingual data during pretraining can significantly reduce the dependency on bilingual data. Conversely, the direct application of bilingual data for fine-tuning is more resource-efficient. The size of parallel data collected by LexMatcher is considerably smaller than that of mixed monolingual data, and the training process is only a single stage.  

Table 7: Compound translation error rate (CTER) on CoGnition. Instance and Aggregate denote the instancelevel and aggregate-level compound translation error rates, respectively.   


<html><body><table><tr><td>Model</td><td>BLEU</td><td>Instance</td><td>Aggregate</td></tr><tr><td>Transformer</td><td>59.5</td><td>28.4</td><td>62.9</td></tr><tr><td>Transformer+CReg</td><td>61.3</td><td>20.2</td><td>48.3</td></tr><tr><td>LLaMA2-ICL</td><td>38.9</td><td>68.6</td><td>87.4</td></tr><tr><td>LLaMA2-SFT</td><td>62.4</td><td>18.5</td><td>43.9</td></tr><tr><td>LexMatcher</td><td>63.5</td><td>15.6</td><td>37.3</td></tr></table></body></html>  

### 5.5  Compositional Generalization  

The data with balanced atom distribution can enhance the performance of compositional generalization, and we verify it on CoGnition (Li et al., 2021). The evaluation metrics include instancelevel CTER which denotes the translation accuracy of novel compounds, and aggregate-level CTER which measures the translation consistency across different contexts. We use the sense retrieval of LexMatcher to obtain 70,272 parallel sentences from the full training data (196,246) with $K{=}50$ For LLM, we apply ICL with 8 examples and finetune LLaMA2-7B on the randomly sampled training data, of which the size is similar to the retrieved data. The results are shown in Table 7. ICL does not yield good compositional generalization performance, while the fine-tuned LLaMA2 outperforms the previous NMT models significantly. LexMatcher achieves lower compound translation error rates than SFT with the same amount of training data, demonstrating the positive effect of the more balanced data distribution.  

## 6 Conclusion  

In this paper, we present LexMatcher, a dictionarycentric data collection method for supervised finetuning, and make open-source LLMs a better translation model. We use the bilingual dictionary as the pivot and try to collect limited parallel sentence pairs to cover the senses uniformly. Experiments and analyses validate the effectiveness of LexMatcher from multiple perspectives including zero-shot translation, disambiguation, and terminology translation. One potential avenue for future research involves extending LexMatcher to lowresource scenarios, where the utilization of monolingual data is crucial for achieving satisfactory translation performance.  

## 7 Limitations  

This work focuses solely on improving translation performance for medium and high-resource language pairs. For low-resource language pairs that inherently lack parallel data, it is crucial to explore how to optimize LLMs on such translation tasks by integrating dictionaries, monolingual, and possible bilingual data.  

## References  

Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2022. Incontext examples selection for machine translation. CoRR, abs/2212.02437.  

Marjan Ghazvininejad, Hila Gonen, and Luke Zettlemoyer. 2023. Dictionary-based phrase-level prompting of large language models for machine translation. CoRR, abs/2302.07856.  

Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2023. Incontext examples selection for machine translation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8857-8873, Toronto, Canada. Association for Computational Linguistics.  

Mitchell A Gordon, Kevin Duh, and Jared Kaplan. 2021. Data and parameter scaling laws for neural machine translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5915-5922, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.  

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. 2023. Textbooks are all you need. CoRR, abs/2306.11644.  

Francis Bond, Piek Vossen, John McCrae, and Christiane Fellbaum. 2016. CILI: the collaborative interlingual index. In Proceedings of the 8th Global WordNet Conference (GWC), pages 50-57, Bucharest, Romania. Global Wordnet Association.  

Niccolo Campolungo, Federico Martelli, Francesco Saina, and Roberto Navigli. 2022. DiBiMT: A novel benchmark for measuring Word Sense Disambiguation biases in Machine Translation. In Proceedings of the 6Oth Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4331-4352, Dublin, Ireland. Association for Computational Linguistics.  

Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018. Understanding back-translation at scale. In Proceedings of the 2018 Conference on Empirical Methods inNatural LanguageProcessing, Brussels, Belgium, October 31 - November 4, 2018, pages 489-500. Association for Computational Linguistics.  

Patrick Fernandes, Behrooz Ghorbani, Xavier Garcia, Markus Freitag, and Orhan Firat. 2023. Scaling laws for multilingual neural machine translation. In InternationalConferenceonMachineLearning,ICML 2023,23-29July2023,Honolulu,Hawaii,USA,volume 202 of Proceedings of Machine Learning Research,pages 10053-10071.PMLR.  

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. How good are GPT models at machine translation? A comprehensive evaluation. CoRR, abs/2302.09210.  

Junjie Hu, Hiroaki Hayashi, Kyunghyun Cho, and Graham Neubig. 2022. DEEP: DEnoising entity pretraining for neural machine translation. In Proceedings of the60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1753-1766, Dublin, Ireland. Association for Computational Linguistics.  

Vivek Iyer, Pinzhen Chen, and Alexandra Birch. 2023. Towards effective disambiguation for machine translation with large language models. In Proceedings of theEighth Conference on MachineTranslation, pages 482-495, Singapore. Association for Computational Linguistics.  

Wenxiang Jiao, Jen-tse Huang, Wenxuan Wang, Zhiwei He, Tian Liang, Xing Wang, Shuming Shi, and Zhaopeng Tu. 2023. ParroT: Translating during chat using large language models tuned with human translation and feedback. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15009-15020, Singapore. Association for Computational Linguistics.  

Gaurav Kumar, George Foster, Colin Cherry, and Maxim Krikun. 2019. Reinforcement learning based curriculum optimization for neural machine translation. In Proceedings of the 2019 Conference of theNorthAmericanChapteroftheAssociationfor Computational Linguistics:HumanLanguageTechnologies, Volume 1 (Long and Short Papers), pages 2054-2061, Minneapolis, Minnesota. Association for Computational Linguistics.  

Yafu Li, Yongjing Yin, Yulong Chen, and Yue Zhang. 2021. On compositional generalization of neural machine translation. In Proceedings of the 59th Annual  

Meeting of the Association for Computational Linguistics and the11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4767-4780, Online. Association for Computational Linguistics.  

Yafu Li, Yongjing Yin, Jing Li, and Yue Zhang. 2022. Prompt-driven neural machine translation. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2579-2590, Dublin, Ireland. Association for Computational Linguistics.  

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023. Textbooks are all you need II: phi-1.5 technical report. CoRR, abs/2309.05463.  

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona T. Diab, Veselin Stoyanov, and Xian Li. 2022. Few-shot learning with multilingual generative language models. In EMNLP 2022, pages 9019-9052.  

Hongyuan Lu, Haoyang Huang, Dongdong Zhang, Haoran Yang, Wai Lam, and Furu Wei. 2023. Chainof-dictionary prompting elicits translation in large language models. CoRR, abs/2305.06575.  

Zhuoyuan Mao and Yen Yu. 2024. Tuning llms with contrastive alignment instructions for machine translation in unseen, low-resource languages. CoRR, abs/2401.05811.  

Tasnim Mohiuddin, Philipp Koehn, Vishrav Chaudhary, James Cross, Shruti Bhosale, and Shafiq Joty. 2022. Data selection curriculum for neural machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1569-1582, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.  

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40thAnnual Meetingof theAssociationfor Computational Linguistics, pages 311-318. Association for Computational Linguistics.  

Arkil Patel, Satwik Bhattamishra, Phil Blunsom, and Navin Goyal. 2022. Revisiting the compositional generalization abilities of neural sequence models. In Proceedings of the 6Oth Annual Meeting of the AssociationforComputational Linguistics(Volume 2: Short Papers), pages 424-434, Dublin, Ireland. Association for Computational Linguistics.  

Ricardo Rei, Jose G. C. de Souza, Duarte M. Alves, Chrysoula Zerva, Ana C. Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, and André F. T. Martins. 2022. COMET-22: unbabel-ist 2022 submission for the metrics shared task. In Proceedings of the SeventhConference onMachineTranslation,WMT  

2022,Abu Dhabi, United Arab Emirates (Hybrid) December 7-8, 2022, pages 578-585. Association for Computational Linguistics.  

Kirill Semenov, Vilém Zouhar, Tom Kocmi, Dongdong Zhang, Wangchunshu Zhou, and Yuchen Eleanor Jiang. 2023. Findings of the wmt 2023 shared task on machine translation with terminologies. In ProceedingsoftheEightConferenceonMachineTranslation (WMT). Association for Computational Linguistics.  

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Improving neural machine translation models with monolingual data. In Proceedings of the 54th Annual Meeting of the Association forComputational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics.  

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.  

Marlies van der Wees, Arianna Bisazza, and Christof Monz. 2017. Dynamic data selection for neural machine translation. In Proceedings of the 2017 Conference onEmpirical Methods inNatural LanguageProcessing, pages 1400-1410, Copenhagen, Denmark. Association for Computational Linguistics.  

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, undefinedukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pages 6000-6010. Curran Associates Inc.  

Shuo Wang, Zhixing Tan, and Yang Liu. 2022. Integrating vectorized lexical constraints for neural machine translation.In Proceedings of the 60th Annual Meetingof theAssociation for Computational Linguistics (Volume 1: Long Papers), pages 7063-7073, Dublin, Ireland. Association for Computational Linguistics.  

Wei Wang, Taro Watanabe, Macduff Hughes, Tetsuji Nakagawa, and Ciprian Chelba. 2018. Denoising neural machine translation training with trusted data and online data selection. In Proceedings of the Third Conference on Machine Translation:Research Papers, pages 133-143, Brussels, Belgium. Association for Computational Linguistics.  

Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, Tianxiang Hu, Shangjie Li, Binyuan Hui, Bowen Yu, Dayiheng Liu, Baosong Yang, Fei Huang, and Jun Xie. 2023. Polylm: An open source polyglot large language model. CoRR, abs/2307.06018.  

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2024. A paradigm shift in machine translation: Boosting translation performance of large language models. In ICLR.  

Wen Yang, Chong Li, Jiajun Zhang, and Chengqing Zong. 2023. Bigtrans: Augmenting large language models with multilingual translation capability over 100 languages. CoRR, abs/2305.18098.  

Jiali Zeng, Fandong Meng, Yongjing Yin, and Jie Zhou. 2024. TIM: teaching large language models to translate with comparison. In AAAI2024, pages 19488- 19496. AAAI Press.  

Zixin Zeng, Rui Wang, Yichong Leng, Junliang Guo, Shufang Xie, Xu Tan, Tao Qin, and Tie-Yan Liu. 2023. Extract and attend: Improving entity translation in neural machine translation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 1697-1710, Toronto, Canada. Association for Computational Linguistics.  

Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng. 2023. Bayling: Bridging cross-lingual alignment and instruction following through interactive translation for large language models. CoRR, abs/2306.10968.  

Yang Zhao, Jiajun Zhang, Yu Zhou, and Chengqing Zong. 2020. Knowledge graphs enhanced neural machine translation. In Proceedings of the TwentyNinthInternational Joint Conference on Artificial Intelligence, IJCAI 2020, pages 4039-4045. ijcai.org.  

Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang. 2023. Multilingual machine translation with large language models: Empirical results and analysis. CoRR, abs/2304.04675.  

## A  Computational Details  

We conducted experiments using the Huggingface Transformers of version 4.29. The experiments are performed on NVIDIA A100 GPU, and all the results are run once with the random seed 42. According to the data license of WMT22, the data released for the General MT task can be freely used for research purposes.  

## B Prompts Used for Manipulating ChatGPT and Terminology Translation  

The prompt used to manipulate ChatGPT consists of three parts (Figure 6 (a)). The first part is used to describe the task: generate a pair of parallel sentences in two languages, which can reflect the meaning of a given segment pair accurately. The second part is a specific example to demonstrate the format of the input and output including a segment pair, a definition of the sense, and parallel sentences. The third part is the segment pair.  

The prompt used for terminology translation is shown in Figure 6 (b), and the constraint translation is prepended to the instruction.  

## C  Corpus Preprocessing  

Since the fltered data of Russian $\Leftrightarrow$ English is significantly less than the other language pairs, we introduce the training set from Tatoeba translation challenge $2021^{14}$ . We filter data with the commonly used rule-based methods and model-based QE. The rules include the following categories: (1) sentence-level deduplication, (2) filter out the sentences longer than 1o0 words or contain a single word exceeding 40 characters, (3) remove sentence pairs where the ratio of source sentence length to target sentence length is significantly different, i.e., below 1/3 or above 3, (4) filter out the sentences with high repeat ratio, i.e., the proportion of the frequency of the most frequent word in a sentence to the total word frequency greater than 0.3, and (5) filter out the sentences in which the proportion of the content words is between 0.3 and 0.8. In this way, low-quality data can be efficiently filtered out, saving time and resources for the subsequent model-based QE.  

We utilize one of the state-of-the-art QE models, COMET-KIW15, to obtain sentence-level quality scores. For every sentence pair in the training data, we calculate the QE score for the translation from English to the foreign language. These scores are utilized for both translation directions, as evaluating both directions of the training data can be computationally expensive. We remove sentence pairs with low data quality, e.g., those that have a score below 40. We use spaCy16 for the lemmatization.  

### D Ablation Study  

The detailed results of ablation experiments and the investigation of fine-tuning ALMA are presented in Table 8.  

Given a pair of words that are of the same meaning but in different languages, and the definition of the meaning, please generate a pair of sentences in English and Chinese respectively, which can reflect the meaning most accurately.  

Example:  

Word Pair: English:“head”- Chinese:“负责人"   
Definition: the person in charge of a group of people or an organization   
Sentence pair: English: She resigned as head of department. Chinese:她辞去了部门负责人的职务。   
Now, please generate three sentence pairs for the below word pair:   
Word Pair: English: “being” - Chinese:“生物   
Definition: a living thing that has (or can develop) the ability to act or function independently   
SentencePairs:  

Table 8: The detailed results of ablation study.   


<html><body><table><tr><td>Model</td><td>Zh=En BLEU/COMET</td><td>En=Zh BLEU/COMET</td><td>De=En BLEU/COMET</td><td>En=De BLEU/COMET</td><td>Ru=En BLEU/COMET</td><td>En=Ru BLEU/COMET</td></tr><tr><td>Dev</td><td>23.59/78.94</td><td>35.43/84.28</td><td>29.04/83.63</td><td>28.58/84.09</td><td>36.68/83.58</td><td>24.23/85.54</td></tr><tr><td>+Supplement</td><td>23.69/79.05</td><td>36.50/84.20</td><td>29.45/83.82</td><td>28.67/83.98</td><td>38.03/83.80</td><td>25.14/85.49</td></tr><tr><td>+Retrieval</td><td>25.36/79.46</td><td>40.14/86.01</td><td>32.37/84.31</td><td>33.26/85.77</td><td>40.86/84.36</td><td>29.00/87.03</td></tr><tr><td>LexMatcher</td><td>24.81/79.13</td><td>40.34/86.11</td><td>32.33/84.29</td><td>33.56/86.31</td><td>41.01/84.43</td><td>28.97/87.23</td></tr></table></body></html>  