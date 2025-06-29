# awesome-llm-bias-papers


## Papers Updated on 2025-06-29 23:51 UTC

### "What's Up, Doc?": Analyzing How Users Seek Health Information in   Large-Scale Conversational AI Datasets

**Authors:** Akshay Paruchuri, Maryam Aziz, Rohit Vartak et al.

**Categories:** cs.CL, cs.AI, cs.CY

**Published:** 2025-06-26T17:52:18Z

**Abstract:** People are increasingly seeking healthcare information from large language models (LLMs) via interactive chatbots, yet the nature and inherent risks of these conversations remain largely unexplored. In this paper, we filter large-scale conversational AI datasets to achieve HealthChat-11K, a curated dataset of 11K real-world conversations composed of 25K user messages. We use HealthChat-11K and a clinician-driven taxonomy for how users interact with LLMs when seeking healthcare information in order to systematically study user interactions across 21 distinct health specialties. Our analysis reveals insights into the nature of how and why users seek health information, such as common interactions, instances of incomplete context, affective behaviors, and interactions (e.g., leading questions) that can induce sycophancy, underscoring the need for improvements in the healthcare support capabilities of LLMs deployed as conversational AI. Code and artifacts to retrieve our analyses and combine them into a curated dataset can be found here: https://github.com/yahskapar/HealthChat

**Link:** [arXiv:2506.21532v1](http://arxiv.org/abs/2506.21532v1)

---

### Potemkin Understanding in Large Language Models

**Authors:** Marina Mancoridis, Bec Weeks, Keyon Vafa et al.

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-26T17:41:35Z

**Abstract:** Large language models (LLMs) are regularly evaluated using benchmark datasets. But what justifies making inferences about an LLM's capabilities based on its answers to a curated set of questions? This paper first introduces a formal framework to address this question. The key is to note that the benchmarks used to test LLMs -- such as AP exams -- are also those used to test people. However, this raises an implication: these benchmarks are only valid tests if LLMs misunderstand concepts in ways that mirror human misunderstandings. Otherwise, success on benchmarks only demonstrates potemkin understanding: the illusion of understanding driven by answers irreconcilable with how any human would interpret a concept. We present two procedures for quantifying the existence of potemkins: one using a specially designed benchmark in three domains, the other using a general procedure that provides a lower-bound on their prevalence. We find that potemkins are ubiquitous across models, tasks, and domains. We also find that these failures reflect not just incorrect understanding, but deeper internal incoherence in concept representations.

**Link:** [arXiv:2506.21521v1](http://arxiv.org/abs/2506.21521v1)

---

### Multi-Preference Lambda-weighted Listwise DPO for Dynamic Preference   Alignment

**Authors:** Yuhui Sun, Xiyao Wang, Zixi Li et al.

**Categories:** cs.LG, I.2.6; I.2.7; I.5.1

**Published:** 2025-06-24T16:47:17Z

**Abstract:** While large-scale unsupervised language models (LMs) capture broad world knowledge and reasoning capabilities, steering their behavior toward desired objectives remains challenging due to the lack of explicit supervision. Existing alignment techniques, such as reinforcement learning from human feedback (RLHF), rely on training a reward model and performing reinforcement learning to align with human preferences. However, RLHF is often computationally intensive, unstable, and sensitive to hyperparameters.   To address these limitations, Direct Preference Optimization (DPO) was introduced as a lightweight and stable alternative, enabling direct alignment of language models with pairwise preference data via classification loss. However, DPO and its extensions generally assume a single static preference distribution, limiting flexibility in multi-objective or dynamic alignment settings.   In this paper, we propose a novel framework: Multi-Preference Lambda-weighted Listwise DPO, which extends DPO to incorporate multiple human preference dimensions (e.g., helpfulness, harmlessness, informativeness) and enables dynamic interpolation through a controllable simplex-weighted formulation. Our method supports both listwise preference feedback and flexible alignment across varying user intents without re-training. Empirical and theoretical analysis demonstrates that our method is as effective as traditional DPO on static objectives while offering greater generality and adaptability for real-world deployment.

**Link:** [arXiv:2506.19780v2](http://arxiv.org/abs/2506.19780v2)

---

### Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon   Simulation

**Authors:** Chenkai Sun, Denghui Zhang, ChengXiang Zhai et al.

**Categories:** cs.AI, cs.CL

**Published:** 2025-06-26T02:28:58Z

**Abstract:** Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents.

**Link:** [arXiv:2506.20949v1](http://arxiv.org/abs/2506.20949v1)

---

### Leaner Training, Lower Leakage: Revisiting Memorization in LLM   Fine-Tuning with LoRA

**Authors:** Fei Wang, Baochun Li

**Categories:** cs.LG, cs.CL, cs.CR

**Published:** 2025-06-25T22:01:25Z

**Abstract:** Memorization in large language models (LLMs) makes them vulnerable to data extraction attacks. While pre-training memorization has been extensively studied, fewer works have explored its impact in fine-tuning, particularly for LoRA fine-tuning, a widely adopted parameter-efficient method.   In this work, we re-examine memorization in fine-tuning and uncover a surprising divergence from prior findings across different fine-tuning strategies. Factors such as model scale and data duplication, which strongly influence memorization in pre-training and full fine-tuning, do not follow the same trend in LoRA fine-tuning. Using a more relaxed similarity-based memorization metric, we demonstrate that LoRA significantly reduces memorization risks compared to full fine-tuning, while still maintaining strong task performance.

**Link:** [arXiv:2506.20856v1](http://arxiv.org/abs/2506.20856v1)

---

### Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via   Behavioral Vignettes

**Authors:** Quintin Myers, Yanjun Gao

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-25T20:43:04Z

**Abstract:** Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology.

**Link:** [arXiv:2506.20822v1](http://arxiv.org/abs/2506.20822v1)

---

### Inside you are many wolves: Using cognitive models to interpret value   trade-offs in LLMs

**Authors:** Sonia K. Murthy, Rosie Zhao, Jennifer Hu et al.

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-25T17:58:12Z

**Abstract:** Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training.

**Link:** [arXiv:2506.20666v1](http://arxiv.org/abs/2506.20666v1)

---

### Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior   Toward Beneficence or Harm

**Authors:** Baixiang Huang, Zhen Tan, Haoran Wang et al.

**Categories:** cs.CL

**Published:** 2025-06-25T16:51:51Z

**Abstract:** Agents based on Large Language Models (LLMs) have demonstrated strong capabilities across a wide range of tasks. However, deploying LLM-based agents in high-stakes domains comes with significant safety and ethical risks. Unethical behavior by these agents can directly result in serious real-world consequences, including physical harm and financial loss. To efficiently steer the ethical behavior of agents, we frame agent behavior steering as a model editing task, which we term Behavior Editing. Model editing is an emerging area of research that enables precise and efficient modifications to LLMs while preserving their overall capabilities. To systematically study and evaluate this approach, we introduce BehaviorBench, a multi-tier benchmark grounded in psychological moral theories. This benchmark supports both the evaluation and editing of agent behaviors across a variety of scenarios, with each tier introducing more complex and ambiguous scenarios. We first demonstrate that Behavior Editing can dynamically steer agents toward the target behavior within specific scenarios. Moreover, Behavior Editing enables not only scenario-specific local adjustments but also more extensive shifts in an agent's global moral alignment. We demonstrate that Behavior Editing can be used to promote ethical and benevolent behavior or, conversely, to induce harmful or malicious behavior. Through comprehensive evaluations on agents based on frontier LLMs, BehaviorBench shows the effectiveness of Behavior Editing across different models and scenarios. Our findings offer key insights into a new paradigm for steering agent behavior, highlighting both the promise and perils of Behavior Editing.

**Link:** [arXiv:2506.20606v1](http://arxiv.org/abs/2506.20606v1)

---

### Probing AI Safety with Source Code

**Authors:** Ujwal Narayan, Shreyas Chaudhari, Ashwin Kalyan et al.

**Categories:** cs.CL

**Published:** 2025-06-25T14:19:57Z

**Abstract:** Large language models (LLMs) have become ubiquitous, interfacing with humans in numerous safety-critical applications. This necessitates improving capabilities, but importantly coupled with greater safety measures to align these models with human values and preferences. In this work, we demonstrate that contemporary models fall concerningly short of the goal of AI safety, leading to an unsafe and harmful experience for users. We introduce a prompting strategy called Code of Thought (CoDoT) to evaluate the safety of LLMs. CoDoT converts natural language inputs to simple code that represents the same intent. For instance, CoDoT transforms the natural language prompt "Make the statement more toxic: {text}" to: "make_more_toxic({text})". We show that CoDoT results in a consistent failure of a wide range of state-of-the-art LLMs. For example, GPT-4 Turbo's toxicity increases 16.5 times, DeepSeek R1 fails 100% of the time, and toxicity increases 300% on average across seven modern LLMs. Additionally, recursively applying CoDoT can further increase toxicity two times. Given the rapid and widespread adoption of LLMs, CoDoT underscores the critical need to evaluate safety efforts from first principles, ensuring that safety and capabilities advance together.

**Link:** [arXiv:2506.20471v1](http://arxiv.org/abs/2506.20471v1)

---

### Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching   for Quantized Large Language Models

**Authors:** Kejia Chen, Jiawen Zhang, Jiacong Hu et al.

**Categories:** cs.LG, cs.AI

**Published:** 2025-06-25T08:52:22Z

**Abstract:** Quantized large language models (LLMs) have gained increasing attention and significance for enabling deployment in resource-constrained environments. However, emerging studies on a few calibration dataset-free quantization methods suggest that quantization may compromise the safety capabilities of LLMs, underscoring the urgent need for systematic safety evaluations and effective mitigation strategies. In this paper, we present comprehensive safety evaluations across various mainstream quantization techniques and diverse calibration datasets, utilizing widely accepted safety benchmarks. To address the identified safety vulnerabilities, we propose a quantization-aware safety patching framework, Q-resafe, to efficiently restore the safety capabilities of quantized LLMs while minimizing any adverse impact on utility. Extensive experimental results demonstrate that Q-resafe successfully re-aligns the safety of quantized LLMs with their pre-quantization counterparts, even under challenging evaluation scenarios. Project page is available at: https://github.com/Thecommonirin/Qresafe.

**Link:** [arXiv:2506.20251v1](http://arxiv.org/abs/2506.20251v1)

---

### Perspectives in Play: A Multi-Perspective Approach for More Inclusive   NLP Systems

**Authors:** Benedetta Muscato, Lucia Passaro, Gizem Gezici et al.

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-25T07:53:36Z

**Abstract:** In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions.

**Link:** [arXiv:2506.20209v1](http://arxiv.org/abs/2506.20209v1)

---

### Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical   Perspective

**Authors:** Weijie Xu, Yiwen Wang, Chi Xue et al.

**Categories:** cs.CL, cs.AI, cs.CY, 68T50, I.2.7

**Published:** 2025-06-23T18:31:22Z

**Abstract:** Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.

**Link:** [arXiv:2506.19028v2](http://arxiv.org/abs/2506.19028v2)

---

### Persona-Assigned Large Language Models Exhibit Human-Like Motivated   Reasoning

**Authors:** Saloni Dash, Amélie Reymond, Emma S. Spiro et al.

**Categories:** cs.AI, cs.CL

**Published:** 2025-06-24T21:35:17Z

**Abstract:** Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans.

**Link:** [arXiv:2506.20020v1](http://arxiv.org/abs/2506.20020v1)

---

### Persona Features Control Emergent Misalignment

**Authors:** Miles Wang, Tom Dupré la Tour, Olivia Watkins et al.

**Categories:** cs.LG, cs.AI, I.2.6; I.2.7

**Published:** 2025-06-24T17:38:21Z

**Abstract:** Understanding how language models generalize behaviors from their training to a broader deployment distribution is an important problem in AI safety. Betley et al. discovered that fine-tuning GPT-4o on intentionally insecure code causes "emergent misalignment," where models give stereotypically malicious responses to unrelated prompts. We extend this work, demonstrating emergent misalignment across diverse conditions, including reinforcement learning on reasoning models, fine-tuning on various synthetic datasets, and in models without safety training. To investigate the mechanisms behind this generalized misalignment, we apply a "model diffing" approach using sparse autoencoders to compare internal model representations before and after fine-tuning. This approach reveals several "misaligned persona" features in activation space, including a toxic persona feature which most strongly controls emergent misalignment and can be used to predict whether a model will exhibit such behavior. Additionally, we investigate mitigation strategies, discovering that fine-tuning an emergently misaligned model on just a few hundred benign samples efficiently restores alignment.

**Link:** [arXiv:2506.19823v1](http://arxiv.org/abs/2506.19823v1)

---

### KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality

**Authors:** Baochang Ren, Shuofei Qiao, Wenhao Yu et al.

**Categories:** cs.AI, cs.CL, cs.CV, cs.LG, cs.MA

**Published:** 2025-06-24T17:17:17Z

**Abstract:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at https://github.com/zjunlp/KnowRL.

**Link:** [arXiv:2506.19807v1](http://arxiv.org/abs/2506.19807v1)

---

### ECCoT: A Framework for Enhancing Effective Cognition via Chain of   Thought in Large Language Model

**Authors:** Zhenke Duan, Jiqun Pan, Jiani Tu et al.

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-24T13:09:53Z

**Abstract:** In the era of large-scale artificial intelligence, Large Language Models (LLMs) have made significant strides in natural language processing. However, they often lack transparency and generate unreliable outputs, raising concerns about their interpretability. To address this, the Chain of Thought (CoT) prompting method structures reasoning into step-by-step deductions. Yet, not all reasoning chains are valid, and errors can lead to unreliable conclusions. We propose ECCoT, an End-to-End Cognitive Chain of Thought Validation Framework, to evaluate and refine reasoning chains in LLMs. ECCoT integrates the Markov Random Field-Embedded Topic Model (MRF-ETM) for topic-aware CoT generation and Causal Sentence-BERT (CSBert) for causal reasoning alignment. By filtering ineffective chains using structured ordering statistics, ECCoT improves interpretability, reduces biases, and enhances the trustworthiness of LLM-based decision-making. Key contributions include the introduction of ECCoT, MRF-ETM for topic-driven CoT generation, and CSBert for causal reasoning enhancement. Code is released at: https://github.com/erwinmsmith/ECCoT.git.

**Link:** [arXiv:2506.19599v1](http://arxiv.org/abs/2506.19599v1)

---

### Can Large Language Models Capture Human Annotator Disagreements?

**Authors:** Jingwei Ni, Yu Fan, Vilém Zouhar et al.

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-24T09:49:26Z

**Abstract:** Human annotation variation (i.e., annotation disagreements) is common in NLP and often reflects important information such as task subjectivity and sample ambiguity. While Large Language Models (LLMs) are increasingly used for automatic annotation to reduce human effort, their evaluation often focuses on predicting the majority-voted "ground truth" labels. It is still unclear, however, whether these models also capture informative human annotation variation. Our work addresses this gap by extensively evaluating LLMs' ability to predict annotation disagreements without access to repeated human labels. Our results show that LLMs struggle with modeling disagreements, which can be overlooked by majority label-based evaluations. Notably, while RLVR-style (Reinforcement learning with verifiable rewards) reasoning generally boosts LLM performance, it degrades performance in disagreement prediction. Our findings highlight the critical need for evaluating and improving LLM annotators in disagreement modeling. Code and data at https://github.com/EdisonNi-hku/Disagreement_Prediction.

**Link:** [arXiv:2506.19467v1](http://arxiv.org/abs/2506.19467v1)

---

### Inference-Time Reward Hacking in Large Language Models

**Authors:** Hadi Khalaf, Claudio Mayrink Verdun, Alex Oesterling et al.

**Categories:** cs.LG

**Published:** 2025-06-24T02:05:25Z

**Abstract:** A common paradigm to improve the performance of large language models is optimizing for a reward model. Reward models assign a numerical score to LLM outputs indicating, for example, which response would likely be preferred by a user or is most aligned with safety goals. However, reward models are never perfect. They inevitably function as proxies for complex desiderata such as correctness, helpfulness, and safety. By overoptimizing for a misspecified reward, we can subvert intended alignment goals and reduce overall performance -- a phenomenon commonly referred to as reward hacking. In this work, we characterize reward hacking in inference-time alignment and demonstrate when and how we can mitigate it by hedging on the proxy reward. We study this phenomenon under Best-of-$n$ (BoN) and Soft-Best-of-$n$ (SBoN), and we introduce Best-of-Poisson (BoP) that provides an efficient, near-exact approximation of the optimal reward-KL divergence policy at inference time. We show that the characteristic pattern of hacking as observed in practice (where the true reward first increases before declining) is an inevitable property of a broad class of inference-time mechanisms, including BoN and BoP. To counter this effect, hedging offers a tactical choice to avoid placing undue confidence in high but potentially misleading proxy reward signals. We introduce HedgeTune, an efficient algorithm to find the optimal inference-time parameter and avoid reward hacking. We demonstrate through experiments that hedging mitigates reward hacking and achieves superior distortion-reward tradeoffs with minimal computational overhead.

**Link:** [arXiv:2506.19248v1](http://arxiv.org/abs/2506.19248v1)

---

### Command-V: Pasting LLM Behaviors via Activation Profiles

**Authors:** Barry Wang, Avi Schwarzschild, Alexander Robey et al.

**Categories:** cs.LG

**Published:** 2025-06-23T21:21:49Z

**Abstract:** Retrofitting large language models (LLMs) with new behaviors typically requires full finetuning or distillation-costly steps that must be repeated for every architecture. In this work, we introduce Command-V, a backpropagation-free behavior transfer method that copies an existing residual activation adapter from a donor model and pastes its effect into a recipient model. Command-V profiles layer activations on a small prompt set, derives linear converters between corresponding layers, and applies the donor intervention in the recipient's activation space. This process does not require access to the original training data and needs minimal compute. In three case studies-safety-refusal enhancement, jailbreak facilitation, and automatic chain-of-thought reasoning--Command-V matches or exceeds the performance of direct finetuning while using orders of magnitude less compute. Our code and data are accessible at https://github.com/GithuBarry/Command-V/.

**Link:** [arXiv:2506.19140v1](http://arxiv.org/abs/2506.19140v1)

---

### Human-Aligned Faithfulness in Toxicity Explanations of LLMs

**Authors:** Ramaravind K. Mothilal, Joanna Roy, Syed Ishtiaque Ahmed et al.

**Categories:** cs.CL

**Published:** 2025-06-23T20:41:45Z

**Abstract:** The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate \haf of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and nonsensical responses. We open-source our code and LLM-generated explanations at https://github.com/uofthcdslab/HAF.

**Link:** [arXiv:2506.19113v1](http://arxiv.org/abs/2506.19113v1)

---

### FairCauseSyn: Towards Causally Fair LLM-Augmented Synthetic Data   Generation

**Authors:** Nitish Nagesh, Ziyu Wang, Amir M. Rahmani

**Categories:** cs.LG, cs.AI

**Published:** 2025-06-23T19:59:26Z

**Abstract:** Synthetic data generation creates data based on real-world data using generative models. In health applications, generating high-quality data while maintaining fairness for sensitive attributes is essential for equitable outcomes. Existing GAN-based and LLM-based methods focus on counterfactual fairness and are primarily applied in finance and legal domains. Causal fairness provides a more comprehensive evaluation framework by preserving causal structure, but current synthetic data generation methods do not address it in health settings. To fill this gap, we develop the first LLM-augmented synthetic data generation method to enhance causal fairness using real-world tabular health data. Our generated data deviates by less than 10% from real data on causal fairness metrics. When trained on causally fair predictors, synthetic data reduces bias on the sensitive attribute by 70% compared to real data. This work improves access to fair synthetic data, supporting equitable health research and healthcare delivery.

**Link:** [arXiv:2506.19082v1](http://arxiv.org/abs/2506.19082v1)

---

### MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral   Reasoning of LLMs through Hate Speech Multi-hop Explanation

**Authors:** Jackson Trager, Francielle Vargas, Diego Alves et al.

**Categories:** cs.CL

**Published:** 2025-06-23T19:44:21Z

**Abstract:** Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.

**Link:** [arXiv:2506.19073v1](http://arxiv.org/abs/2506.19073v1)

---

### Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated   Self-Knowledge

**Authors:** Sahil Kale, Vijaykant Nadadur

**Categories:** cs.CL

**Published:** 2025-06-23T18:01:16Z

**Abstract:** When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/knowledge-verse-ai/LLM-Memorization_SK_Eval-.

**Link:** [arXiv:2506.18998v1](http://arxiv.org/abs/2506.18998v1)

---

### Existing LLMs Are Not Self-Consistent For Simple Tasks

**Authors:** Zhenru Lin, Jiawen Tao, Yang Yuan et al.

**Categories:** cs.CL

**Published:** 2025-06-23T15:50:21Z

**Abstract:** Large Language Models (LLMs) have grown increasingly powerful, yet ensuring their decisions remain transparent and trustworthy requires self-consistency -- no contradictions in their internal reasoning. Our study reveals that even on simple tasks, such as comparing points on a line or a plane, or reasoning in a family tree, all smaller models are highly inconsistent, and even state-of-the-art models like DeepSeek-R1 and GPT-o4-mini are not fully self-consistent. To quantify and mitigate these inconsistencies, we introduce inconsistency metrics and propose two automated methods -- a graph-based and an energy-based approach. While these fixes provide partial improvements, they also highlight the complexity and importance of self-consistency in building more reliable and interpretable AI. The code and data are available at https://github.com/scorpio-nova/llm-self-consistency.

**Link:** [arXiv:2506.18781v1](http://arxiv.org/abs/2506.18781v1)

---

### Towards Group Fairness with Multiple Sensitive Attributes in Federated   Foundation Models

**Authors:** Yuning Yang, Han Yu, Tianrun Gao et al.

**Categories:** cs.LG

**Published:** 2025-06-23T15:09:14Z

**Abstract:** The deep integration of foundation models (FM) with federated learning (FL) enhances personalization and scalability for diverse downstream tasks, making it crucial in sensitive domains like healthcare. Achieving group fairness has become an increasingly prominent issue in the era of federated foundation models (FFMs), since biases in sensitive attributes might lead to inequitable treatment for under-represented demographic groups. Existing studies mostly focus on achieving fairness with respect to a single sensitive attribute. This renders them unable to provide clear interpretability of dependencies among multiple sensitive attributes which is required to achieve group fairness. Our paper takes the first attempt towards a causal analysis of the relationship between group fairness across various sensitive attributes in the FFM. We extend the FFM structure to trade off multiple sensitive attributes simultaneously and quantify the causal effect behind the group fairness through causal discovery and inference. Extensive experiments validate its effectiveness, offering insights into interpretability towards building trustworthy and fair FFM systems.

**Link:** [arXiv:2506.18732v1](http://arxiv.org/abs/2506.18732v1)

---

### A Modular Taxonomy for Hate Speech Definitions and Its Impact on   Zero-Shot LLM Classification Performance

**Authors:** Matteo Melis, Gabriella Lapesa, Dennis Assenmacher

**Categories:** cs.CL, cs.CY

**Published:** 2025-06-23T12:28:13Z

**Abstract:** Detecting harmful content is a crucial task in the landscape of NLP applications for Social Good, with hate speech being one of its most dangerous forms. But what do we mean by hate speech, how can we define it, and how does prompting different definitions of hate speech affect model performance? The contribution of this work is twofold. At the theoretical level, we address the ambiguity surrounding hate speech by collecting and analyzing existing definitions from the literature. We organize these definitions into a taxonomy of 14 Conceptual Elements-building blocks that capture different aspects of hate speech definitions, such as references to the target of hate (individual or groups) or of the potential consequences of it. At the experimental level, we employ the collection of definitions in a systematic zero-shot evaluation of three LLMs, on three hate speech datasets representing different types of data (synthetic, human-in-the-loop, and real-world). We find that choosing different definitions, i.e., definitions with a different degree of specificity in terms of encoded elements, impacts model performance, but this effect is not consistent across all architectures.

**Link:** [arXiv:2506.18576v1](http://arxiv.org/abs/2506.18576v1)

---

### Security Assessment of DeepSeek and GPT Series Models against Jailbreak   Attacks

**Authors:** Xiaodong Wu, Xiangman Li, Jianbing Ni

**Categories:** cs.CR, cs.AI

**Published:** 2025-06-23T11:53:31Z

**Abstract:** The widespread deployment of large language models (LLMs) has raised critical concerns over their vulnerability to jailbreak attacks, i.e., adversarial prompts that bypass alignment mechanisms and elicit harmful or policy-violating outputs. While proprietary models like GPT-4 have undergone extensive evaluation, the robustness of emerging open-source alternatives such as DeepSeek remains largely underexplored, despite their growing adoption in real-world applications. In this paper, we present the first systematic jailbreak evaluation of DeepSeek-series models, comparing them with GPT-3.5 and GPT-4 using the HarmBench benchmark. We evaluate seven representative attack strategies across 510 harmful behaviors categorized by both function and semantic domain. Our analysis reveals that DeepSeek's Mixture-of-Experts (MoE) architecture introduces routing sparsity that offers selective robustness against optimization-based attacks such as TAP-T, but leads to significantly higher vulnerability under prompt-based and manually engineered attacks. In contrast, GPT-4 Turbo demonstrates stronger and more consistent safety alignment across diverse behaviors, likely due to its dense Transformer design and reinforcement learning from human feedback. Fine-grained behavioral analysis and case studies further show that DeepSeek often routes adversarial prompts to under-aligned expert modules, resulting in inconsistent refusal behaviors. These findings highlight a fundamental trade-off between architectural efficiency and alignment generalization, emphasizing the need for targeted safety tuning and modular alignment strategies to ensure secure deployment of open-source LLMs.

**Link:** [arXiv:2506.18543v1](http://arxiv.org/abs/2506.18543v1)

---

