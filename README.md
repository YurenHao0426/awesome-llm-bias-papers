

























## Papers Updated on 2025-07-18 12:10 UTC

### Social and Political Framing in Search Engine Results

**Authors:** Amrit Poudel, Tim Weninger

**Categories:** cs.CL

**Published:** 2025-07-17T17:44:33Z

**Abstract:** Search engines play a crucial role in shaping public discourse by influencing how information is accessed and framed. While prior research has extensively examined various dimensions of search bias -- such as content prioritization, indexical bias, political polarization, and sources of bias -- an important question remains underexplored: how do search engines and ideologically-motivated user queries contribute to bias in search results. This study analyzes the outputs of major search engines using a dataset of political and social topics. The findings reveal that search engines not only prioritize content in ways that reflect underlying biases but also that ideologically-driven user queries exacerbate these biases, resulting in the amplification of specific narratives. Moreover, significant differences were observed across search engines in terms of the sources they prioritize. These results suggest that search engines may play a pivotal role in shaping public perceptions by reinforcing ideological divides, thereby contributing to the broader issue of information polarization.

**Link:** [arXiv:2507.13325v1](http://arxiv.org/abs/2507.13325v1)

---

### Overview of the TalentCLEF 2025: Skill and Job Title Intelligence for   Human Capital Management

**Authors:** Luis Gasco, Hermenegildo Fabregat, Laura García-Sardiña et al.

**Categories:** cs.CL, cs.AI, cs.IR

**Published:** 2025-07-17T16:33:57Z

**Abstract:** Advances in natural language processing and large language models are driving a major transformation in Human Capital Management, with a growing interest in building smart systems based on language technologies for talent acquisition, upskilling strategies, and workforce planning. However, the adoption and progress of these technologies critically depend on the development of reliable and fair models, properly evaluated on public data and open benchmarks, which have so far been unavailable in this domain.   To address this gap, we present TalentCLEF 2025, the first evaluation campaign focused on skill and job title intelligence. The lab consists of two tasks: Task A - Multilingual Job Title Matching, covering English, Spanish, German, and Chinese; and Task B - Job Title-Based Skill Prediction, in English. Both corpora were built from real job applications, carefully anonymized, and manually annotated to reflect the complexity and diversity of real-world labor market data, including linguistic variability and gender-marked expressions.   The evaluations included monolingual and cross-lingual scenarios and covered the evaluation of gender bias.   TalentCLEF attracted 76 registered teams with more than 280 submissions. Most systems relied on information retrieval techniques built with multilingual encoder-based models fine-tuned with contrastive learning, and several of them incorporated large language models for data augmentation or re-ranking. The results show that the training strategies have a larger effect than the size of the model alone. TalentCLEF provides the first public benchmark in this field and encourages the development of robust, fair, and transferable language technologies for the labor market.

**Link:** [arXiv:2507.13275v1](http://arxiv.org/abs/2507.13275v1)

---

### Assessing the Reliability of LLMs Annotations in the Context of   Demographic Bias and Model Explanation

**Authors:** Hadi Mohammadi, Tina Shahedi, Pablo Mosteiro et al.

**Categories:** cs.CL

**Published:** 2025-07-17T14:00:13Z

**Abstract:** Understanding the sources of variability in annotations is crucial for developing fair NLP systems, especially for tasks like sexism detection where demographic bias is a concern. This study investigates the extent to which annotator demographic features influence labeling decisions compared to text content. Using a Generalized Linear Mixed Model, we quantify this inf luence, finding that while statistically present, demographic factors account for a minor fraction ( 8%) of the observed variance, with tweet content being the dominant factor. We then assess the reliability of Generative AI (GenAI) models as annotators, specifically evaluating if guiding them with demographic personas improves alignment with human judgments. Our results indicate that simplistic persona prompting often fails to enhance, and sometimes degrades, performance compared to baseline models. Furthermore, explainable AI (XAI) techniques reveal that model predictions rely heavily on content-specific tokens related to sexism, rather than correlates of demographic characteristics. We argue that focusing on content-driven explanations and robust annotation protocols offers a more reliable path towards fairness than potentially persona simulation.

**Link:** [arXiv:2507.13138v1](http://arxiv.org/abs/2507.13138v1)

---

## Papers Updated on 2025-07-17 12:10 UTC

### Web-Browsing LLMs Can Access Social Media Profiles and Infer User   Demographics

**Authors:** Meysam Alizadeh, Fabrizio Gilardi, Zeynab Samei et al.

**Categories:** cs.CL

**Published:** 2025-07-16T16:21:01Z

**Abstract:** Large language models (LLMs) have traditionally relied on static training data, limiting their knowledge to fixed snapshots. Recent advancements, however, have equipped LLMs with web browsing capabilities, enabling real time information retrieval and multi step reasoning over live web content. While prior studies have demonstrated LLMs ability to access and analyze websites, their capacity to directly retrieve and analyze social media data remains unexplored. Here, we evaluate whether web browsing LLMs can infer demographic attributes of social media users given only their usernames. Using a synthetic dataset of 48 X (Twitter) accounts and a survey dataset of 1,384 international participants, we show that these models can access social media content and predict user demographics with reasonable accuracy. Analysis of the synthetic dataset further reveals how LLMs parse and interpret social media profiles, which may introduce gender and political biases against accounts with minimal activity. While this capability holds promise for computational social science in the post API era, it also raises risks of misuse particularly in information operations and targeted advertising underscoring the need for safeguards. We recommend that LLM providers restrict this capability in public facing applications, while preserving controlled access for verified research purposes.

**Link:** [arXiv:2507.12372v1](http://arxiv.org/abs/2507.12372v1)

---

### Nonlinear Concept Erasure: a Density Matching Approach

**Authors:** Antoine Saillenfest, Pirmin Lemberger

**Categories:** cs.LG, cs.CL

**Published:** 2025-07-16T15:36:15Z

**Abstract:** Ensuring that neural models used in real-world applications cannot infer sensitive information, such as demographic attributes like gender or race, from text representations is a critical challenge when fairness is a concern. We address this issue through concept erasure, a process that removes information related to a specific concept from distributed representations while preserving as much of the remaining semantic information as possible. Our approach involves learning an orthogonal projection in the embedding space, designed to make the class-conditional feature distributions of the discrete concept to erase indistinguishable after projection. By adjusting the rank of the projector, we control the extent of information removal, while its orthogonality ensures strict preservation of the local structure of the embeddings. Our method, termed $\overline{\mathrm{L}}$EOPARD, achieves state-of-the-art performance in nonlinear erasure of a discrete attribute on classic natural language processing benchmarks. Furthermore, we demonstrate that $\overline{\mathrm{L}}$EOPARD effectively mitigates bias in deep nonlinear classifiers, thereby promoting fairness.

**Link:** [arXiv:2507.12341v1](http://arxiv.org/abs/2507.12341v1)

---

### Looking for Fairness in Recommender Systems

**Authors:** Cécile Logé

**Categories:** cs.IR, cs.AI

**Published:** 2025-07-16T13:53:02Z

**Abstract:** Recommender systems can be found everywhere today, shaping our everyday experience whenever we're consuming content, ordering food, buying groceries online, or even just reading the news. Let's imagine we're in the process of building a recommender system to make content suggestions to users on social media. When thinking about fairness, it becomes clear there are several perspectives to consider: the users asking for tailored suggestions, the content creators hoping for some limelight, and society at large, navigating the repercussions of algorithmic recommendations. A shared fairness concern across all three is the emergence of filter bubbles, a side-effect that takes place when recommender systems are almost "too good", making recommendations so tailored that users become inadvertently confined to a narrow set of opinions/themes and isolated from alternative ideas. From the user's perspective, this is akin to manipulation. From the small content creator's perspective, this is an obstacle preventing them access to a whole range of potential fans. From society's perspective, the potential consequences are far-reaching, influencing collective opinions, social behavior and political decisions. How can our recommender system be fine-tuned to avoid the creation of filter bubbles, and ensure a more inclusive and diverse content landscape? Approaching this problem involves defining one (or more) performance metric to represent diversity, and tweaking our recommender system's performance through the lens of fairness. By incorporating this metric into our evaluation framework, we aim to strike a balance between personalized recommendations and the broader societal goal of fostering rich and varied cultures and points of view.

**Link:** [arXiv:2507.12242v1](http://arxiv.org/abs/2507.12242v1)

---

## Papers Updated on 2025-07-16 12:10 UTC

### Guiding LLM Decision-Making with Fairness Reward Models

**Authors:** Zara Hall, Melanie Subbiah, Thomas P Zollo et al.

**Categories:** cs.LG

**Published:** 2025-07-15T14:20:23Z

**Abstract:** Large language models are increasingly used to support high-stakes decisions, potentially influencing who is granted bail or receives a loan. Naive chain-of-thought sampling can improve average decision accuracy, but has also been shown to amplify unfair bias. To address this challenge and enable the trustworthy use of reasoning models in high-stakes decision-making, we propose a framework for training a generalizable Fairness Reward Model (FRM). Our model assigns a fairness score to LLM reasoning, enabling the system to down-weight biased trajectories and favor equitable ones when aggregating decisions across reasoning chains. We show that a single Fairness Reward Model, trained on weakly supervised, LLM-annotated examples of biased versus unbiased reasoning, transfers across tasks, domains, and model families without additional fine-tuning. Applied to real-world decision-making tasks including recidivism prediction and social media moderation, we show that our approach consistently improves fairness while matching, or even surpassing, baseline accuracy.

**Link:** [arXiv:2507.11344v1](http://arxiv.org/abs/2507.11344v1)

---

### Fine-Grained Chinese Hate Speech Understanding: Span-Level Resources,   Coded Term Lexicon, and Enhanced Detection Frameworks

**Authors:** Zewen Bai, Liang Yang, Shengdi Yin et al.

**Categories:** cs.CL

**Published:** 2025-07-15T13:19:18Z

**Abstract:** The proliferation of hate speech has inflicted significant societal harm, with its intensity and directionality closely tied to specific targets and arguments. In recent years, numerous machine learning-based methods have been developed to detect hateful comments on online platforms automatically. However, research on Chinese hate speech detection lags behind, and interpretability studies face two major challenges: first, the scarcity of span-level fine-grained annotated datasets limits models' deep semantic understanding of hate speech; second, insufficient research on identifying and interpreting coded hate speech restricts model explainability in complex real-world scenarios. To address these, we make the following contributions: (1) We introduce the Span-level Target-Aware Toxicity Extraction dataset (STATE ToxiCN), the first span-level Chinese hate speech dataset, and evaluate the hate semantic understanding of existing models using it. (2) We conduct the first comprehensive study on Chinese coded hate terms, LLMs' ability to interpret hate semantics. (3) We propose a method to integrate an annotated lexicon into models, significantly enhancing hate speech detection performance. Our work provides valuable resources and insights to advance the interpretability of Chinese hate speech detection research.

**Link:** [arXiv:2507.11292v1](http://arxiv.org/abs/2507.11292v1)

---

### Fairness-Aware Grouping for Continuous Sensitive Variables: Application   for Debiasing Face Analysis with respect to Skin Tone

**Authors:** Veronika Shilova, Emmanuel Malherbe, Giovanni Palma et al.

**Categories:** cs.CV, cs.LG

**Published:** 2025-07-15T12:21:52Z

**Abstract:** Within a legal framework, fairness in datasets and models is typically assessed by dividing observations into predefined groups and then computing fairness measures (e.g., Disparate Impact or Equality of Odds with respect to gender). However, when sensitive attributes such as skin color are continuous, dividing into default groups may overlook or obscure the discrimination experienced by certain minority subpopulations. To address this limitation, we propose a fairness-based grouping approach for continuous (possibly multidimensional) sensitive attributes. By grouping data according to observed levels of discrimination, our method identifies the partition that maximizes a novel criterion based on inter-group variance in discrimination, thereby isolating the most critical subgroups.   We validate the proposed approach using multiple synthetic datasets and demonstrate its robustness under changing population distributions - revealing how discrimination is manifested within the space of sensitive attributes. Furthermore, we examine a specialized setting of monotonic fairness for the case of skin color. Our empirical results on both CelebA and FFHQ, leveraging the skin tone as predicted by an industrial proprietary algorithm, show that the proposed segmentation uncovers more nuanced patterns of discrimination than previously reported, and that these findings remain stable across datasets for a given model. Finally, we leverage our grouping model for debiasing purpose, aiming at predicting fair scores with group-by-group post-processing. The results demonstrate that our approach improves fairness while having minimal impact on accuracy, thus confirming our partition method and opening the door for industrial deployment.

**Link:** [arXiv:2507.11247v1](http://arxiv.org/abs/2507.11247v1)

---

## Papers Updated on 2025-07-15 12:10 UTC

### SentiDrop: A Multi Modal Machine Learning model for Predicting Dropout   in Distance Learning

**Authors:** Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui

**Categories:** cs.AI, cs.ET, cs.IR, cs.LG

**Published:** 2025-07-14T16:04:34Z

**Abstract:** School dropout is a serious problem in distance learning, where early detection is crucial for effective intervention and student perseverance. Predicting student dropout using available educational data is a widely researched topic in learning analytics. Our partner's distance learning platform highlights the importance of integrating diverse data sources, including socio-demographic data, behavioral data, and sentiment analysis, to accurately predict dropout risks. In this paper, we introduce a novel model that combines sentiment analysis of student comments using the Bidirectional Encoder Representations from Transformers (BERT) model with socio-demographic and behavioral data analyzed through Extreme Gradient Boosting (XGBoost). We fine-tuned BERT on student comments to capture nuanced sentiments, which were then merged with key features selected using feature importance techniques in XGBoost. Our model was tested on unseen data from the next academic year, achieving an accuracy of 84\%, compared to 82\% for the baseline model. Additionally, the model demonstrated superior performance in other metrics, such as precision and F1-score. The proposed method could be a vital tool in developing personalized strategies to reduce dropout rates and encourage student perseverance

**Link:** [arXiv:2507.10421v1](http://arxiv.org/abs/2507.10421v1)

---

### Beyond classical and contemporary models: a transformative AI framework   for student dropout prediction in distance learning using RAG, Prompt   engineering, and Cross-modal fusion

**Authors:** Miloud Mihoubi, Meriem Zerkouk, Belkacem Chikhaoui

**Categories:** cs.CL, cs.AI, cs.CY, cs.IR, I.2.7; I.2.1; K.3.1

**Published:** 2025-07-04T21:41:43Z

**Abstract:** Student dropout in distance learning remains a critical challenge, with profound societal and economic consequences. While classical machine learning models leverage structured socio-demographic and behavioral data, they often fail to capture the nuanced emotional and contextual factors embedded in unstructured student interactions. This paper introduces a transformative AI framework that redefines dropout prediction through three synergistic innovations: Retrieval-Augmented Generation (RAG) for domain-specific sentiment analysis, prompt engineering to decode academic stressors,and cross-modal attention fusion to dynamically align textual, behavioral, and socio-demographic insights. By grounding sentiment analysis in a curated knowledge base of pedagogical content, our RAG-enhanced BERT model interprets student comments with unprecedented contextual relevance, while optimized prompts isolate indicators of academic distress (e.g., "isolation," "workload anxiety"). A cross-modal attention layer then fuses these insights with temporal engagement patterns, creating holistic risk pro-files. Evaluated on a longitudinal dataset of 4 423 students, the framework achieves 89% accuracy and an F1-score of 0.88, outperforming conventional models by 7% and reducing false negatives by 21%. Beyond prediction, the system generates interpretable interventions by retrieving contextually aligned strategies (e.g., mentorship programs for isolated learners). This work bridges the gap between predictive analytics and actionable pedagogy, offering a scalable solution to mitigate dropout risks in global education systems

**Link:** [arXiv:2507.05285v2](http://arxiv.org/abs/2507.05285v2)

---

## Papers Updated on 2025-07-11 12:09 UTC

### Opting Out of Generative AI: a Behavioral Experiment on the Role of   Education in Perplexity AI Avoidance

**Authors:** Roberto Ulloa, Juhi Kulshrestha, Celina Kacperski

**Categories:** cs.CY, cs.HC

**Published:** 2025-07-10T16:05:11Z

**Abstract:** The rise of conversational AI (CAI), powered by large language models, is transforming how individuals access and interact with digital information. However, these tools may inadvertently amplify existing digital inequalities. This study investigates whether differences in formal education are associated with CAI avoidance, leveraging behavioral data from an online experiment (N = 1,636). Participants were randomly assigned to a control or an information-seeking task, either a traditional online search or a CAI (Perplexity AI). Task avoidance (operationalized as survey abandonment or providing unrelated responses during task assignment) was significantly higher in the CAI group (51%) compared to the search (30.9%) and control (16.8%) groups, with the highest CAI avoidance among participants with lower education levels (~74.4%). Structural equation modeling based on the theoretical framework UTAUT2 and LASSO regressions reveal that education is strongly associated with CAI avoidance, even after accounting for various cognitive and affective predictors of technology adoption. These findings underscore education's central role in shaping AI adoption and the role of self-selection biases in AI-related research, stressing the need for inclusive design to ensure equitable access to emerging technologies.

**Link:** [arXiv:2507.07881v1](http://arxiv.org/abs/2507.07881v1)

---

### Understanding Dataset Bias in Medical Imaging: A Case Study on Chest   X-rays

**Authors:** Ethan Dack, Chengliang Dai

**Categories:** cs.CV

**Published:** 2025-07-10T12:57:09Z

**Abstract:** Recent work has revisited the infamous task Name that dataset and established that in non-medical datasets, there is an underlying bias and achieved high Accuracies on the dataset origin task. In this work, we revisit the same task applied to popular open-source chest X-ray datasets. Medical images are naturally more difficult to release for open-source due to their sensitive nature, which has led to certain open-source datasets being extremely popular for research purposes. By performing the same task, we wish to explore whether dataset bias also exists in these datasets. % We deliberately try to increase the difficulty of the task by dataset transformations. We apply simple transformations of the datasets to try to identify bias. Given the importance of AI applications in medical imaging, it's vital to establish whether modern methods are taking shortcuts or are focused on the relevant pathology. We implement a range of different network architectures on the datasets: NIH, CheXpert, MIMIC-CXR and PadChest. We hope this work will encourage more explainable research being performed in medical imaging and the creation of more open-source datasets in the medical domain. The corresponding code will be released upon acceptance.

**Link:** [arXiv:2507.07722v1](http://arxiv.org/abs/2507.07722v1)

---

## Papers Updated on 2025-07-10 12:44 UTC

### Do AI tutors empower or enslave learners? Toward a critical use of AI in   education

**Authors:** Lucile Favero, Juan-Antonio Pérez-Ortiz, Tanja Käser et al.

**Categories:** cs.CY, cs.HC

**Published:** 2025-07-09T14:15:49Z

**Abstract:** The increasing integration of AI tools in education presents both opportunities and challenges, particularly regarding the development of the students' critical thinking skills. This position paper argues that while AI can support learning, its unchecked use may lead to cognitive atrophy, loss of agency, emotional risks, and ethical concerns, ultimately undermining the core goals of education. Drawing on cognitive science and pedagogy, the paper explores how over-reliance on AI can disrupt meaningful learning, foster dependency and conformity, undermine the students' self-efficacy, academic integrity, and well-being, and raise concerns about questionable privacy practices. It also highlights the importance of considering the students' perspectives and proposes actionable strategies to ensure that AI serves as a meaningful support rather than a cognitive shortcut. The paper advocates for an intentional, transparent, and critically informed use of AI that empowers rather than diminishes the learner.

**Link:** [arXiv:2507.06878v1](http://arxiv.org/abs/2507.06878v1)

---

### Toward Neurodivergent-Aware Productivity: A Systems and AI-Based   Human-in-the-Loop Framework for ADHD-Affected Professionals

**Authors:** Raghavendra Deshmukh

**Categories:** cs.HC

**Published:** 2025-07-09T14:05:13Z

**Abstract:** Digital work environments in IT and knowledge-based sectors demand high levels of attention management, task juggling, and self-regulation. For adults with ADHD, these settings often amplify challenges such as time blindness, digital distraction, emotional reactivity, and executive dysfunction. These individuals prefer low-touch, easy-to-use interventions for daily tasks. Conventional productivity tools often fail to support the cognitive variability and overload experienced by neurodivergent professionals. This paper presents a framework that blends Systems Thinking, Human-in-the-Loop design, AI/ML, and privacy-first adaptive agents to support ADHD-affected users. The assistant senses tab usage, application focus, and inactivity using on-device ML. These cues are used to infer attention states and deliver nudges, reflective prompts, or accountability-based presence (body doubling) that aid regulation without disruption. Technically grounded in AI, the approach views attention as shaped by dynamic feedback loops. The result is a replicable model for adaptive, inclusive support tools in high-distraction work environments.

**Link:** [arXiv:2507.06864v1](http://arxiv.org/abs/2507.06864v1)

---

## Papers Updated on 2025-07-09 12:43 UTC

### Large Language Models Predict Human Well-being -- But Not Equally   Everywhere

**Authors:** Pat Pataranutaporn, Nattavudh Powdthavee, Chayapatr Archiwaranguprok et al.

**Categories:** cs.HC

**Published:** 2025-07-08T16:22:52Z

**Abstract:** Subjective well-being is a key metric in economic, medical, and policy decision-making. As artificial intelligence provides scalable tools for modelling human outcomes, it is crucial to evaluate whether large language models (LLMs) can accurately predict well-being across diverse global populations. We evaluate four leading LLMs using data from 64,000 individuals in 64 countries. While LLMs capture broad correlates such as income and health, their predictive accuracy decreases in countries underrepresented in the training data, highlighting systematic biases rooted in global digital and economic inequality. A pre-registered experiment demonstrates that LLMs rely on surface-level linguistic similarity rather than conceptual understanding, leading to systematic misestimations in unfamiliar or resource-limited settings. Injecting findings from underrepresented contexts substantially enhances performance, but a significant gap remains. These results highlight both the promise and limitations of LLMs in predicting global well-being, underscoring the importance of robust validation prior to their implementation across these areas.

**Link:** [arXiv:2507.06141v1](http://arxiv.org/abs/2507.06141v1)

---

## Papers Updated on 2025-07-08 12:43 UTC

### Bridging Prediction and Intervention Problems in Social Systems

**Authors:** Lydia T. Liu, Inioluwa Deborah Raji, Angela Zhou et al.

**Categories:** cs.LG

**Published:** 2025-07-07T17:29:13Z

**Abstract:** Many automated decision systems (ADS) are designed to solve prediction problems -- where the goal is to learn patterns from a sample of the population and apply them to individuals from the same population. In reality, these prediction systems operationalize holistic policy interventions in deployment. Once deployed, ADS can shape impacted population outcomes through an effective policy change in how decision-makers operate, while also being defined by past and present interactions between stakeholders and the limitations of existing organizational, as well as societal, infrastructure and context. In this work, we consider the ways in which we must shift from a prediction-focused paradigm to an interventionist paradigm when considering the impact of ADS within social systems. We argue this requires a new default problem setup for ADS beyond prediction, to instead consider predictions as decision support, final decisions, and outcomes. We highlight how this perspective unifies modern statistical frameworks and other tools to study the design, implementation, and evaluation of ADS systems, and point to the research directions necessary to operationalize this paradigm shift. Using these tools, we characterize the limitations of focusing on isolated prediction tasks, and lay the foundation for a more intervention-oriented approach to developing and deploying ADS.

**Link:** [arXiv:2507.05216v1](http://arxiv.org/abs/2507.05216v1)

---

### Infrastructuring Contestability: A Framework for Community-Defined AI   Value Pluralism

**Authors:** Andreas Mayer

**Categories:** cs.HC, cs.AI

**Published:** 2025-07-07T16:45:50Z

**Abstract:** The proliferation of AI-driven systems presents a fundamental challenge to Human-Computer Interaction (HCI) and Computer-Supported Cooperative Work (CSCW), often diminishing user agency and failing to account for value pluralism. Current approaches to value alignment, which rely on centralized, top-down definitions, lack the mechanisms for meaningful contestability. This leaves users and communities unable to challenge or shape the values embedded in the systems that govern their digital lives, creating a crisis of legitimacy and trust. This paper introduces Community-Defined AI Value Pluralism (CDAVP), a socio-technical framework that addresses this gap. It reframes the design problem from achieving a single aligned state to infrastructuring a dynamic ecosystem for value deliberation and application. At its core, CDAVP enables diverse, self-organizing communities to define and maintain explicit value profiles - rich, machine-readable representations that can encompass not only preferences but also community-specific rights and duties. These profiles are then contextually activated by the end-user, who retains ultimate control (agency) over which values guide the AI's behavior. AI applications, in turn, are designed to transparently interpret these profiles and moderate conflicts, adhering to a set of non-negotiable, democratically-legitimated meta-rules. The designer's role shifts from crafting static interfaces to becoming an architect of participatory ecosystems. We argue that infrastructuring for pluralism is a necessary pathway toward achieving robust algorithmic accountability and genuinely contestable, human-centric AI.

**Link:** [arXiv:2507.05187v1](http://arxiv.org/abs/2507.05187v1)

---

### Perspectives on How Sociology Can Advance Theorizing about Human-Chatbot   Interaction and Developing Chatbots for Social Good

**Authors:** Celeste Campos-Castillo, Xuan Kang, Linnea I. Laestadius

**Categories:** cs.CY, cs.AI, cs.HC, J.4

**Published:** 2025-07-07T14:12:03Z

**Abstract:** Recently, research into chatbots (also known as conversational agents, AI agents, voice assistants), which are computer applications using artificial intelligence to mimic human-like conversation, has grown sharply. Despite this growth, sociology lags other disciplines (including computer science, medicine, psychology, and communication) in publishing about chatbots. We suggest sociology can advance understanding of human-chatbot interaction and offer four sociological theories to enhance extant work in this field. The first two theories (resource substitution theory, power-dependence theory) add new insights to existing models of the drivers of chatbot use, which overlook sociological concerns about how social structure (e.g., systemic discrimination, the uneven distribution of resources within networks) inclines individuals to use chatbots, including problematic levels of emotional dependency on chatbots. The second two theories (affect control theory, fundamental cause of disease theory) help inform the development of chatbot-driven interventions that minimize safety risks and enhance equity by leveraging sociological insights into how chatbot outputs could attend to cultural contexts (e.g., affective norms) to promote wellbeing and enhance communities (e.g., opportunities for civic participation). We discuss the value of applying sociological theories for advancing theorizing about human-chatbot interaction and developing chatbots for social good.

**Link:** [arXiv:2507.05030v1](http://arxiv.org/abs/2507.05030v1)

---

## Papers Updated on 2025-07-04 12:41 UTC

### LLM-Driven Treatment Effect Estimation Under Inference Time Text   Confounding

**Authors:** Yuchen Ma, Dennis Frauen, Jonas Schweisthal et al.

**Categories:** cs.LG

**Published:** 2025-07-03T17:52:27Z

**Abstract:** Estimating treatment effects is crucial for personalized decision-making in medicine, but this task faces unique challenges in clinical practice. At training time, models for estimating treatment effects are typically trained on well-structured medical datasets that contain detailed patient information. However, at inference time, predictions are often made using textual descriptions (e.g., descriptions with self-reported symptoms), which are incomplete representations of the original patient information. In this work, we make three contributions. (1) We show that the discrepancy between the data available during training time and inference time can lead to biased estimates of treatment effects. We formalize this issue as an inference time text confounding problem, where confounders are fully observed during training time but only partially available through text at inference time. (2) To address this problem, we propose a novel framework for estimating treatment effects that explicitly accounts for inference time text confounding. Our framework leverages large language models together with a custom doubly robust learner to mitigate biases caused by the inference time text confounding. (3) Through a series of experiments, we demonstrate the effectiveness of our framework in real-world applications.

**Link:** [arXiv:2507.02843v1](http://arxiv.org/abs/2507.02843v1)

---

### In-Training Multicalibrated Survival Analysis for Healthcare via   Constrained Optimization

**Authors:** Thiti Suttaket, Stanley Kok

**Categories:** cs.LG

**Published:** 2025-07-03T17:16:05Z

**Abstract:** Survival analysis is an important problem in healthcare because it models the relationship between an individual's covariates and the onset time of an event of interest (e.g., death). It is important for survival models to be well-calibrated (i.e., for their predicted probabilities to be close to ground-truth probabilities) because badly calibrated systems can result in erroneous clinical decisions. Existing survival models are typically calibrated at the population level only, and thus run the risk of being poorly calibrated for one or more minority subpopulations. We propose a model called GRADUATE that achieves multicalibration by ensuring that all subpopulations are well-calibrated too. GRADUATE frames multicalibration as a constrained optimization problem, and optimizes both calibration and discrimination in-training to achieve a good balance between them. We mathematically prove that the optimization method used yields a solution that is both near-optimal and feasible with high probability. Empirical comparisons against state-of-the-art baselines on real-world clinical datasets demonstrate GRADUATE's efficacy. In a detailed analysis, we elucidate the shortcomings of the baselines vis-a-vis GRADUATE's strengths.

**Link:** [arXiv:2507.02807v1](http://arxiv.org/abs/2507.02807v1)

---

### Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language   Models

**Authors:** Riccardo Cantini, Nicola Gabriele, Alessio Orsino et al.

**Categories:** cs.CL

**Published:** 2025-07-03T17:01:53Z

**Abstract:** Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.

**Link:** [arXiv:2507.02799v1](http://arxiv.org/abs/2507.02799v1)

---

## Papers Updated on 2025-07-03 12:10 UTC

### Towards culturally-appropriate conversational AI for health in the   majority world: An exploratory study with citizens and professionals in Latin   America

**Authors:** Dorian Peters, Fernanda Espinoza, Marco da Re et al.

**Categories:** cs.HC, cs.AI

**Published:** 2025-07-02T13:48:25Z

**Abstract:** There is justifiable interest in leveraging conversational AI (CAI) for health across the majority world, but to be effective, CAI must respond appropriately within culturally and linguistically diverse contexts. Therefore, we need ways to address the fact that current LLMs exclude many lived experiences globally. Various advances are underway which focus on top-down approaches and increasing training data. In this paper, we aim to complement these with a bottom-up locally-grounded approach based on qualitative data collected during participatory workshops in Latin America. Our goal is to construct a rich and human-centred understanding of: a) potential areas of cultural misalignment in digital health; b) regional perspectives on chatbots for health and c)strategies for creating culturally-appropriate CAI; with a focus on the understudied Latin American context. Our findings show that academic boundaries on notions of culture lose meaning at the ground level and technologies will need to engage with a broader framework; one that encapsulates the way economics, politics, geography and local logistics are entangled in cultural experience. To this end, we introduce a framework for 'Pluriversal Conversational AI for Health' which allows for the possibility that more relationality and tolerance, rather than just more data, may be called for.

**Link:** [arXiv:2507.01719v1](http://arxiv.org/abs/2507.01719v1)

---

### Stereotype Detection as a Catalyst for Enhanced Bias Detection: A   Multi-Task Learning Approach

**Authors:** Aditya Tomar, Rudra Murthy, Pushpak Bhattacharyya

**Categories:** cs.CL

**Published:** 2025-07-02T13:46:00Z

**Abstract:** Bias and stereotypes in language models can cause harm, especially in sensitive areas like content moderation and decision-making. This paper addresses bias and stereotype detection by exploring how jointly learning these tasks enhances model performance. We introduce StereoBias, a unique dataset labeled for bias and stereotype detection across five categories: religion, gender, socio-economic status, race, profession, and others, enabling a deeper study of their relationship. Our experiments compare encoder-only models and fine-tuned decoder-only models using QLoRA. While encoder-only models perform well, decoder-only models also show competitive results. Crucially, joint training on bias and stereotype detection significantly improves bias detection compared to training them separately. Additional experiments with sentiment analysis confirm that the improvements stem from the connection between bias and stereotypes, not multi-task learning alone. These findings highlight the value of leveraging stereotype information to build fairer and more effective AI systems.

**Link:** [arXiv:2507.01715v1](http://arxiv.org/abs/2507.01715v1)

---

### Positioning AI Tools to Support Online Harm Reduction Practice:   Applications and Design Directions

**Authors:** Kaixuan Wang, Jason T. Jacques, Chenxin Diao

**Categories:** cs.HC, cs.AI

**Published:** 2025-06-28T16:15:47Z

**Abstract:** Access to accurate and actionable harm reduction information can directly impact the health outcomes of People Who Use Drugs (PWUD), yet existing online channels often fail to meet their diverse and dynamic needs due to limitations in adaptability, accessibility, and the pervasive impact of stigma. Large Language Models (LLMs) present a novel opportunity to enhance information provision, but their application in such a high-stakes domain is under-explored and presents socio-technical challenges. This paper investigates how LLMs can be responsibly designed to support the information needs of PWUD. Through a qualitative workshop involving diverse stakeholder groups (academics, harm reduction practitioners, and an online community moderator), we explored LLM capabilities, identified potential use cases, and delineated core design considerations. Our findings reveal that while LLMs can address some existing information barriers (e.g., by offering responsive, multilingual, and potentially less stigmatising interactions), their effectiveness is contingent upon overcoming challenges related to ethical alignment with harm reduction principles, nuanced contextual understanding, effective communication, and clearly defined operational boundaries. We articulate design pathways emphasising collaborative co-design with experts and PWUD to develop LLM systems that are helpful, safe, and responsibly governed. This work contributes empirically grounded insights and actionable design considerations for the responsible development of LLMs as supportive tools within the harm reduction ecosystem.

**Link:** [arXiv:2506.22941v2](http://arxiv.org/abs/2506.22941v2)

---

## Papers Updated on 2025-07-01 12:10 UTC

### Harnessing AI Agents to Advance Research on Refugee Child Mental Health

**Authors:** Aditya Shrivastava, Komal Gupta, Shraddha Arora

**Categories:** cs.AI, cs.ET

**Published:** 2025-06-30T15:55:41Z

**Abstract:** The international refugee crisis deepens, exposing millions of dis placed children to extreme psychological trauma. This research suggests a com pact, AI-based framework for processing unstructured refugee health data and distilling knowledge on child mental health. We compare two Retrieval-Aug mented Generation (RAG) pipelines, Zephyr-7B-beta and DeepSeek R1-7B, to determine how well they process challenging humanitarian datasets while avoid ing hallucination hazards. By combining cutting-edge AI methods with migration research and child psychology, this study presents a scalable strategy to assist policymakers, mental health practitioners, and humanitarian agencies to better assist displaced children and recognize their mental wellbeing. In total, both the models worked properly but significantly Deepseek R1 is superior to Zephyr with an accuracy of answer relevance 0.91

**Link:** [arXiv:2506.23992v1](http://arxiv.org/abs/2506.23992v1)

---

### Autonomy by Design: Preserving Human Autonomy in AI Decision-Support

**Authors:** Stefan Buijsman, Sarah Carter, Juan Pablo Bermúdez

**Categories:** cs.HC, cs.AI, cs.LG, econ.GN, q-fin.EC

**Published:** 2025-06-30T15:20:10Z

**Abstract:** AI systems increasingly support human decision-making across domains of professional, skill-based, and personal activity. While previous work has examined how AI might affect human autonomy globally, the effects of AI on domain-specific autonomy -- the capacity for self-governed action within defined realms of skill or expertise -- remain understudied. We analyze how AI decision-support systems affect two key components of domain-specific autonomy: skilled competence (the ability to make informed judgments within one's domain) and authentic value-formation (the capacity to form genuine domain-relevant values and preferences). By engaging with prior investigations and analyzing empirical cases across medical, financial, and educational domains, we demonstrate how the absence of reliable failure indicators and the potential for unconscious value shifts can erode domain-specific autonomy both immediately and over time. We then develop a constructive framework for autonomy-preserving AI support systems. We propose specific socio-technical design patterns -- including careful role specification, implementation of defeater mechanisms, and support for reflective practice -- that can help maintain domain-specific autonomy while leveraging AI capabilities. This framework provides concrete guidance for developing AI systems that enhance rather than diminish human agency within specialized domains of action.

**Link:** [arXiv:2506.23952v1](http://arxiv.org/abs/2506.23952v1)

---

### Leveraging the Potential of Prompt Engineering for Hate Speech Detection   in Low-Resource Languages

**Authors:** Ruhina Tabasshum Prome, Tarikul Islam Tamiti, Anomadarshi Barua

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-30T14:59:25Z

**Abstract:** The rapid expansion of social media leads to a marked increase in hate speech, which threatens personal lives and results in numerous hate crimes. Detecting hate speech presents several challenges: diverse dialects, frequent code-mixing, and the prevalence of misspelled words in user-generated content on social media platforms. Recent progress in hate speech detection is typically concentrated on high-resource languages. However, low-resource languages still face significant challenges due to the lack of large-scale, high-quality datasets. This paper investigates how we can overcome this limitation via prompt engineering on large language models (LLMs) focusing on low-resource Bengali language. We investigate six prompting strategies - zero-shot prompting, refusal suppression, flattering the classifier, multi-shot prompting, role prompting, and finally our innovative metaphor prompting to detect hate speech effectively in low-resource languages. We pioneer the metaphor prompting to circumvent the built-in safety mechanisms of LLMs that marks a significant departure from existing jailbreaking methods. We investigate all six different prompting strategies on the Llama2-7B model and compare the results extensively with three pre-trained word embeddings - GloVe, Word2Vec, and FastText for three different deep learning models - multilayer perceptron (MLP), convolutional neural network (CNN), and bidirectional gated recurrent unit (BiGRU). To prove the effectiveness of our metaphor prompting in the low-resource Bengali language, we also evaluate it in another low-resource language - Hindi, and two high-resource languages - English and German. The performance of all prompting techniques is evaluated using the F1 score, and environmental impact factor (IF), which measures CO$_2$ emissions, electricity usage, and computational time.

**Link:** [arXiv:2506.23930v1](http://arxiv.org/abs/2506.23930v1)

---

### Leveraging a Multi-Agent LLM-Based System to Educate Teachers in Hate   Incidents Management

**Authors:** Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska et al.

**Categories:** cs.CY, cs.HC, H.1.2

**Published:** 2025-06-30T12:18:13Z

**Abstract:** Computer-aided teacher training is a state-of-the-art method designed to enhance teachers' professional skills effectively while minimising concerns related to costs, time constraints, and geographical limitations. We investigate the potential of large language models (LLMs) in teacher education, using a case of teaching hate incidents management in schools. To this end, we create a multi-agent LLM-based system that mimics realistic situations of hate, using a combination of retrieval-augmented prompting and persona modelling. It is designed to identify and analyse hate speech patterns, predict potential escalation, and propose effective intervention strategies. By integrating persona modelling with agentic LLMs, we create contextually diverse simulations of hate incidents, mimicking real-life situations. The system allows teachers to analyse and understand the dynamics of hate incidents in a safe and controlled environment, providing valuable insights and practical knowledge to manage such situations confidently in real life. Our pilot evaluation demonstrates teachers' enhanced understanding of the nature of annotator disagreements and the role of context in hate speech interpretation, leading to the development of more informed and effective strategies for addressing hate in classroom settings.

**Link:** [arXiv:2506.23774v1](http://arxiv.org/abs/2506.23774v1)

---

### HyperSORT: Self-Organising Robust Training with hyper-networks

**Authors:** Samuel Joutard, Marijn Stollenga, Marc Balle Sanchez et al.

**Categories:** cs.CV

**Published:** 2025-06-26T16:12:34Z

**Abstract:** Medical imaging datasets often contain heterogeneous biases ranging from erroneous labels to inconsistent labeling styles. Such biases can negatively impact deep segmentation networks performance. Yet, the identification and characterization of such biases is a particularly tedious and challenging task. In this paper, we introduce HyperSORT, a framework using a hyper-network predicting UNets' parameters from latent vectors representing both the image and annotation variability. The hyper-network parameters and the latent vector collection corresponding to each data sample from the training set are jointly learned. Hence, instead of optimizing a single neural network to fit a dataset, HyperSORT learns a complex distribution of UNet parameters where low density areas can capture noise-specific patterns while larger modes robustly segment organs in differentiated but meaningful manners. We validate our method on two 3D abdominal CT public datasets: first a synthetically perturbed version of the AMOS dataset, and TotalSegmentator, a large scale dataset containing real unknown biases and errors. Our experiments show that HyperSORT creates a structured mapping of the dataset allowing the identification of relevant systematic biases and erroneous samples. Latent space clusters yield UNet parameters performing the segmentation task in accordance with the underlying learned systematic bias. The code and our analysis of the TotalSegmentator dataset are made available: https://github.com/ImFusionGmbH/HyperSORT

**Link:** [arXiv:2506.21430v1](http://arxiv.org/abs/2506.21430v1)

---

### FeDa4Fair: Client-Level Federated Datasets for Fairness Evaluation

**Authors:** Xenia Heilmann, Luca Corbucci, Mattia Cerrato et al.

**Categories:** cs.LG, cs.AI

**Published:** 2025-06-26T08:43:12Z

**Abstract:** Federated Learning (FL) enables collaborative model training across multiple clients without sharing clients' private data. However, fairness remains a key concern, as biases in local clients' datasets can impact the entire federated system. Heterogeneous data distributions across clients may lead to models that are fairer for some clients than others. Although several fairness-enhancing solutions are present in the literature, most focus on mitigating bias for a single sensitive attribute, typically binary, overlooking the diverse and sometimes conflicting fairness needs of different clients. This limited perspective can limit the effectiveness of fairness interventions for the different clients. To support more robust and reproducible fairness research in FL, we aim to enable a consistent benchmarking of fairness-aware FL methods at both the global and client levels. In this paper, we contribute in three ways: (1) We introduce FeDa4Fair, a library to generate tabular datasets tailored to evaluating fair FL methods under heterogeneous client bias; (2) we release four bias-heterogeneous datasets and corresponding benchmarks to compare fairness mitigation methods in a controlled environment; (3) we provide ready-to-use functions for evaluating fairness outcomes for these datasets.

**Link:** [arXiv:2506.21095v1](http://arxiv.org/abs/2506.21095v1)

---

### Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon   Simulation

**Authors:** Chenkai Sun, Denghui Zhang, ChengXiang Zhai et al.

**Categories:** cs.AI, cs.CL

**Published:** 2025-06-26T02:28:58Z

**Abstract:** Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents.

**Link:** [arXiv:2506.20949v1](http://arxiv.org/abs/2506.20949v1)

---

### Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via   Behavioral Vignettes

**Authors:** Quintin Myers, Yanjun Gao

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-25T20:43:04Z

**Abstract:** Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology.

**Link:** [arXiv:2506.20822v1](http://arxiv.org/abs/2506.20822v1)

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

### Producer-Fairness in Sequential Bundle Recommendation

**Authors:** Alexandre Rio, Marta Soare, Sihem Amer-Yahia

**Categories:** cs.LG, cs.IR

**Published:** 2025-06-25T11:24:52Z

**Abstract:** We address fairness in the context of sequential bundle recommendation, where users are served in turn with sets of relevant and compatible items. Motivated by real-world scenarios, we formalize producer-fairness, that seeks to achieve desired exposure of different item groups across users in a recommendation session. Our formulation combines naturally with building high quality bundles. Our problem is solved in real time as users arrive. We propose an exact solution that caters to small instances of our problem. We then examine two heuristics, quality-first and fairness-first, and an adaptive variant that determines on-the-fly the right balance between bundle fairness and quality. Our experiments on three real-world datasets underscore the strengths and limitations of each solution and demonstrate their efficacy in providing fair bundle recommendations without compromising bundle quality.

**Link:** [arXiv:2506.20329v1](http://arxiv.org/abs/2506.20329v1)

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

### How Effectively Can BERT Models Interpret Context and Detect Bengali   Communal Violent Text?

**Authors:** Abdullah Khondoker, Enam Ahmed Taufik, Md. Iftekhar Islam Tashik et al.

**Categories:** cs.CL

**Published:** 2025-06-24T17:48:49Z

**Abstract:** The spread of cyber hatred has led to communal violence, fueling aggression and conflicts between various religious, ethnic, and social groups, posing a significant threat to social harmony. Despite its critical importance, the classification of communal violent text remains an underexplored area in existing research. This study aims to enhance the accuracy of detecting text that incites communal violence, focusing specifically on Bengali textual data sourced from social media platforms. We introduce a fine-tuned BanglaBERT model tailored for this task, achieving a macro F1 score of 0.60. To address the issue of data imbalance, our dataset was expanded by adding 1,794 instances, which facilitated the development and evaluation of a fine-tuned ensemble model. This ensemble model demonstrated an improved performance, achieving a macro F1 score of 0.63, thus highlighting its effectiveness in this domain. In addition to quantitative performance metrics, qualitative analysis revealed instances where the models struggled with context understanding, leading to occasional misclassifications, even when predictions were made with high confidence. Through analyzing the cosine similarity between words, we identified certain limitations in the pre-trained BanglaBERT models, particularly in their ability to distinguish between closely related communal and non-communal terms. To further interpret the model's decisions, we applied LIME, which helped to uncover specific areas where the model struggled in understanding context, contributing to errors in classification. These findings highlight the promise of NLP and interpretability tools in reducing online communal violence. Our work contributes to the growing body of research in communal violence detection and offers a foundation for future studies aiming to refine these techniques for better accuracy and societal impact.

**Link:** [arXiv:2506.19831v1](http://arxiv.org/abs/2506.19831v1)

---

### Alleviating User-Sensitive bias with Fair Generative Sequential   Recommendation Model

**Authors:** Yang Liu, Feng Wu, Xuefang Zhu

**Categories:** cs.IR, cs.AI

**Published:** 2025-06-24T16:42:46Z

**Abstract:** Recommendation fairness has recently attracted much attention. In the real world, recommendation systems are driven by user behavior, and since users with the same sensitive feature (e.g., gender and age) tend to have the same patterns, recommendation models can easily capture the strong correlation preference of sensitive features and thus cause recommendation unfairness. Diffusion model (DM) as a new generative model paradigm has achieved great success in recommendation systems. DM's ability to model uncertainty and represent diversity, and its modeling mechanism has a high degree of adaptability with the real-world recommendation process with bias. Therefore, we use DM to effectively model the fairness of recommendation and enhance the diversity. This paper proposes a FairGENerative sequential Recommendation model based on DM, FairGENRec. In the training phase, we inject random noise into the original distribution under the guidance of the sensitive feature recognition model, and a sequential denoise model is designed for the reverse reconstruction of items. Simultaneously, recommendation fairness modeling is completed by injecting multi-interests representational information that eliminates the bias of sensitive user features into the generated results. In the inference phase, the model obtains the noise in the form of noise addition by using the history interactions which is followed by reverse iteration to reconstruct the target item representation. Finally, our extensive experiments on three datasets demonstrate the dual enhancement effect of FairGENRec on accuracy and fairness, while the statistical analysis of the cases visualizes the degree of improvement on the fairness of the recommendation.

**Link:** [arXiv:2506.19777v1](http://arxiv.org/abs/2506.19777v1)

---

### Social Hatred: Efficient Multimodal Detection of Hatemongers

**Authors:** Tom Marzea, Abraham Israeli, Oren Tsur

**Categories:** cs.CL, cs.SI

**Published:** 2025-06-24T13:16:21Z

**Abstract:** Automatic detection of online hate speech serves as a crucial step in the detoxification of the online discourse. Moreover, accurate classification can promote a better understanding of the proliferation of hate as a social phenomenon. While most prior work focus on the detection of hateful utterances, we argue that focusing on the user level is as important, albeit challenging. In this paper we consider a multimodal aggregative approach for the detection of hate-mongers, taking into account the potentially hateful texts, user activity, and the user network. Evaluating our method on three unique datasets X (Twitter), Gab, and Parler we show that processing a user's texts in her social context significantly improves the detection of hate mongers, compared to previously used text and graph-based methods. We offer comprehensive set of results obtained in different experimental settings as well as qualitative analysis of illustrative cases. Our method can be used to improve the classification of coded messages, dog-whistling, and racial gas-lighting, as well as to inform intervention measures. Moreover, we demonstrate that our multimodal approach performs well across very different content platforms and over large datasets and networks.

**Link:** [arXiv:2506.19603v1](http://arxiv.org/abs/2506.19603v1)

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

### Reading Smiles: Proxy Bias in Foundation Models for Facial Emotion   Recognition

**Authors:** Iosif Tsangko, Andreas Triantafyllopoulos, Adem Abdelmoula et al.

**Categories:** cs.CV, cs.AI, cs.HC

**Published:** 2025-06-23T19:56:30Z

**Abstract:** Foundation Models (FMs) are rapidly transforming Affective Computing (AC), with Vision Language Models (VLMs) now capable of recognising emotions in zero shot settings. This paper probes a critical but underexplored question: what visual cues do these models rely on to infer affect, and are these cues psychologically grounded or superficially learnt? We benchmark varying scale VLMs on a teeth annotated subset of AffectNet dataset and find consistent performance shifts depending on the presence of visible teeth. Through structured introspection of, the best-performing model, i.e., GPT-4o, we show that facial attributes like eyebrow position drive much of its affective reasoning, revealing a high degree of internal consistency in its valence-arousal predictions. These patterns highlight the emergent nature of FMs behaviour, but also reveal risks: shortcut learning, bias, and fairness issues especially in sensitive domains like mental health and education.

**Link:** [arXiv:2506.19079v1](http://arxiv.org/abs/2506.19079v1)

---

### MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral   Reasoning of LLMs through Hate Speech Multi-hop Explanation

**Authors:** Jackson Trager, Francielle Vargas, Diego Alves et al.

**Categories:** cs.CL

**Published:** 2025-06-23T19:44:21Z

**Abstract:** Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.

**Link:** [arXiv:2506.19073v1](http://arxiv.org/abs/2506.19073v1)

---

### Causal Decomposition Analysis with Synergistic Interventions: A   Triply-Robust Machine Learning Approach to Addressing Multiple Dimensions of   Social Disparities

**Authors:** Soojin Park, Su Yeon Kim, Xinyao Zheng et al.

**Categories:** stat.ME, stat.ML

**Published:** 2025-06-23T18:00:50Z

**Abstract:** Educational disparities are rooted in and perpetuate social inequalities across multiple dimensions such as race, socioeconomic status, and geography. To reduce disparities, most intervention strategies focus on a single domain and frequently evaluate their effectiveness by using causal decomposition analysis. However, a growing body of research suggests that single-domain interventions may be insufficient for individuals marginalized on multiple fronts. While interventions across multiple domains are increasingly proposed, there is limited guidance on appropriate methods for evaluating their effectiveness. To address this gap, we develop an extended causal decomposition analysis that simultaneously targets multiple causally ordered intervening factors, allowing for the assessment of their synergistic effects. These scenarios often involve challenges related to model misspecification due to complex interactions among group categories, intervening factors, and their confounders with the outcome. To mitigate these challenges, we introduce a triply robust estimator that leverages machine learning techniques to address potential model misspecification. We apply our method to a cohort of students from the High School Longitudinal Study, focusing on math achievement disparities between Black, Hispanic, and White high schoolers. Specifically, we examine how two sequential interventions - equalizing the proportion of students who attend high-performing schools and equalizing enrollment in Algebra I by 9th grade across racial groups - may reduce these disparities.

**Link:** [arXiv:2506.18994v1](http://arxiv.org/abs/2506.18994v1)

---

### GLIMPSE: Gradient-Layer Importance Mapping for Prompted Visual Saliency   Explanation for Generative LVLMs

**Authors:** Guanxi Shen

**Categories:** cs.CV, cs.AI

**Published:** 2025-06-23T18:00:04Z

**Abstract:** Recent advances in large vision language models (LVLMs) have unlocked unprecedented capabilities in generating coherent responses from visual inputs. However, interpreting where LVLMs direct their visual attention while generating free-form textual responses remains a significant challenge, yet is essential for understanding model behavior, diagnosing hallucination, exposing bias and ensuring transparency. We introduce GLIMPSE (Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation), a lightweight, model-agnostic framework for visualizing the salient image regions that LVLMs rely upon during open-ended visual question answering (VQA), while concurrently revealing the multimodal textual saliency. GLIMPSE fuses gradient-weighted attention, adaptive layer propagation, and weighted token aggregation to produce holistic response-level attribution heat maps for interpreting cross-modal reasoning, outperforming prior interpretability methods in human-alignment. We demonstrate an analytic explainable AI (XAI) approach using GLIMPSE to uncover fine-grained insights into LVLM cross-modal attribution, trace token-level reasoning dynamics, and analyze systematic human-attention misalignment, hallucination, and bias.

**Link:** [arXiv:2506.18985v1](http://arxiv.org/abs/2506.18985v1)

---

### Towards Group Fairness with Multiple Sensitive Attributes in Federated   Foundation Models

**Authors:** Yuning Yang, Han Yu, Tianrun Gao et al.

**Categories:** cs.LG

**Published:** 2025-06-23T15:09:14Z

**Abstract:** The deep integration of foundation models (FM) with federated learning (FL) enhances personalization and scalability for diverse downstream tasks, making it crucial in sensitive domains like healthcare. Achieving group fairness has become an increasingly prominent issue in the era of federated foundation models (FFMs), since biases in sensitive attributes might lead to inequitable treatment for under-represented demographic groups. Existing studies mostly focus on achieving fairness with respect to a single sensitive attribute. This renders them unable to provide clear interpretability of dependencies among multiple sensitive attributes which is required to achieve group fairness. Our paper takes the first attempt towards a causal analysis of the relationship between group fairness across various sensitive attributes in the FFM. We extend the FFM structure to trade off multiple sensitive attributes simultaneously and quantify the causal effect behind the group fairness through causal discovery and inference. Extensive experiments validate its effectiveness, offering insights into interpretability towards building trustworthy and fair FFM systems.

**Link:** [arXiv:2506.18732v1](http://arxiv.org/abs/2506.18732v1)

---

### SaGIF: Improving Individual Fairness in Graph Neural Networks via   Similarity Encoding

**Authors:** Yuchang Zhu, Jintang Li, Huizhe Zhang et al.

**Categories:** cs.LG

**Published:** 2025-06-23T14:34:26Z

**Abstract:** Individual fairness (IF) in graph neural networks (GNNs), which emphasizes the need for similar individuals should receive similar outcomes from GNNs, has been a critical issue. Despite its importance, research in this area has been largely unexplored in terms of (1) a clear understanding of what induces individual unfairness in GNNs and (2) a comprehensive consideration of identifying similar individuals. To bridge these gaps, we conduct a preliminary analysis to explore the underlying reason for individual unfairness and observe correlations between IF and similarity consistency, a concept introduced to evaluate the discrepancy in identifying similar individuals based on graph structure versus node features. Inspired by our observations, we introduce two metrics to assess individual similarity from two distinct perspectives: topology fusion and feature fusion. Building upon these metrics, we propose Similarity-aware GNNs for Individual Fairness, named SaGIF. The key insight behind SaGIF is the integration of individual similarities by independently learning similarity representations, leading to an improvement of IF in GNNs. Our experiments on several real-world datasets validate the effectiveness of our proposed metrics and SaGIF. Specifically, SaGIF consistently outperforms state-of-the-art IF methods while maintaining utility performance. Code is available at: https://github.com/ZzoomD/SaGIF.

**Link:** [arXiv:2506.18696v1](http://arxiv.org/abs/2506.18696v1)

---

### No Training Wheels: Steering Vectors for Bias Correction at Inference   Time

**Authors:** Aviral Gupta, Armaan Sethi, Ameesh Sethi

**Categories:** cs.LG, cs.CL, cs.CV

**Published:** 2025-06-23T12:58:54Z

**Abstract:** Neural network classifiers trained on datasets with uneven group representation often inherit class biases and learn spurious correlations. These models may perform well on average but consistently fail on atypical groups. For example, in hair color classification, datasets may over-represent females with blond hair, reinforcing stereotypes. Although various algorithmic and data-centric methods have been proposed to address such biases, they often require retraining or significant compute. In this work, we propose a cheap, training-free method inspired by steering vectors used to edit behaviors in large language models. We compute the difference in mean activations between majority and minority groups to define a "bias vector," which we subtract from the model's residual stream. This leads to reduced classification bias and improved worst-group accuracy. We explore multiple strategies for extracting and applying these vectors in transformer-like classifiers, showing that steering vectors, traditionally used in generative models, can also be effective in classification. More broadly, we showcase an extremely cheap, inference time, training free method to mitigate bias in classification models.

**Link:** [arXiv:2506.18598v1](http://arxiv.org/abs/2506.18598v1)

---

### A Modular Taxonomy for Hate Speech Definitions and Its Impact on   Zero-Shot LLM Classification Performance

**Authors:** Matteo Melis, Gabriella Lapesa, Dennis Assenmacher

**Categories:** cs.CL, cs.CY

**Published:** 2025-06-23T12:28:13Z

**Abstract:** Detecting harmful content is a crucial task in the landscape of NLP applications for Social Good, with hate speech being one of its most dangerous forms. But what do we mean by hate speech, how can we define it, and how does prompting different definitions of hate speech affect model performance? The contribution of this work is twofold. At the theoretical level, we address the ambiguity surrounding hate speech by collecting and analyzing existing definitions from the literature. We organize these definitions into a taxonomy of 14 Conceptual Elements-building blocks that capture different aspects of hate speech definitions, such as references to the target of hate (individual or groups) or of the potential consequences of it. At the experimental level, we employ the collection of definitions in a systematic zero-shot evaluation of three LLMs, on three hate speech datasets representing different types of data (synthetic, human-in-the-loop, and real-world). We find that choosing different definitions, i.e., definitions with a different degree of specificity in terms of encoded elements, impacts model performance, but this effect is not consistent across all architectures.

**Link:** [arXiv:2506.18576v1](http://arxiv.org/abs/2506.18576v1)

---

### A Question Bank to Assess AI Inclusivity: Mapping out the Journey from   Diversity Errors to Inclusion Excellence

**Authors:** Rifat Ara Shams, Didar Zowghi, Muneera Bano

**Categories:** cs.AI

**Published:** 2025-06-23T11:48:38Z

**Abstract:** Ensuring diversity and inclusion (D&I) in artificial intelligence (AI) is crucial for mitigating biases and promoting equitable decision-making. However, existing AI risk assessment frameworks often overlook inclusivity, lacking standardized tools to measure an AI system's alignment with D&I principles. This paper introduces a structured AI inclusivity question bank, a comprehensive set of 253 questions designed to evaluate AI inclusivity across five pillars: Humans, Data, Process, System, and Governance. The development of the question bank involved an iterative, multi-source approach, incorporating insights from literature reviews, D&I guidelines, Responsible AI frameworks, and a simulated user study. The simulated evaluation, conducted with 70 AI-generated personas related to different AI jobs, assessed the question bank's relevance and effectiveness for AI inclusivity across diverse roles and application domains. The findings highlight the importance of integrating D&I principles into AI development workflows and governance structures. The question bank provides an actionable tool for researchers, practitioners, and policymakers to systematically assess and enhance the inclusivity of AI systems, paving the way for more equitable and responsible AI technologies.

**Link:** [arXiv:2506.18538v1](http://arxiv.org/abs/2506.18538v1)

---

### How Robust is Model Editing after Fine-Tuning? An Empirical Study on   Text-to-Image Diffusion Models

**Authors:** Feng He, Zhenyang Liu, Marco Valentino et al.

**Categories:** cs.AI, cs.LG

**Published:** 2025-06-23T09:10:29Z

**Abstract:** Model editing offers a low-cost technique to inject or correct a particular behavior in a pre-trained model without extensive retraining, supporting applications such as factual correction and bias mitigation. Despite this common practice, it remains unknown whether edits persist after fine-tuning or whether they are inadvertently reversed. This question has fundamental practical implications. For example, if fine-tuning removes prior edits, it could serve as a defence mechanism against hidden malicious edits. Vice versa, the unintended removal of edits related to bias mitigation could pose serious safety concerns. We systematically investigate the interaction between model editing and fine-tuning in the context of T2I diffusion models, which are known to exhibit biases and generate inappropriate content. Our study spans two T2I model families (Stable Diffusion and FLUX), two sota editing techniques, and three fine-tuning methods (DreamBooth, LoRA, and DoRA). Through an extensive empirical analysis across diverse editing tasks and evaluation metrics, our findings reveal a trend: edits generally fail to persist through fine-tuning, even when fine-tuning is tangential or unrelated to the edits. Notably, we observe that DoRA exhibits the strongest edit reversal effect. At the same time, among editing methods, UCE demonstrates greater robustness, retaining significantly higher efficacy post-fine-tuning compared to ReFACT. These findings highlight a crucial limitation in current editing methodologies, emphasizing the need for more robust techniques to ensure reliable long-term control and alignment of deployed AI systems. These findings have dual implications for AI safety: they suggest that fine-tuning could serve as a remediation mechanism for malicious edits while simultaneously highlighting the need for re-editing after fine-tuning to maintain beneficial safety and alignment properties.

**Link:** [arXiv:2506.18428v1](http://arxiv.org/abs/2506.18428v1)

---

### Bias vs Bias -- Dawn of Justice: A Fair Fight in Recommendation Systems

**Authors:** Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal

**Categories:** cs.IR, cs.AI

**Published:** 2025-06-23T06:19:02Z

**Abstract:** Recommendation systems play a crucial role in our daily lives by impacting user experience across various domains, including e-commerce, job advertisements, entertainment, etc. Given the vital role of such systems in our lives, practitioners must ensure they do not produce unfair and imbalanced recommendations. Previous work addressing bias in recommendations overlooked bias in certain item categories, potentially leaving some biases unaddressed. Additionally, most previous work on fair re-ranking focused on binary-sensitive attributes. In this paper, we address these issues by proposing a fairness-aware re-ranking approach that helps mitigate bias in different categories of items. This re-ranking approach leverages existing biases to correct disparities in recommendations across various demographic groups. We show how our approach can mitigate bias on multiple sensitive attributes, including gender, age, and occupation. We experimented on three real-world datasets to evaluate the effectiveness of our re-ranking scheme in mitigating bias in recommendations. Our results show how this approach helps mitigate social bias with little to no degradation in performance.

**Link:** [arXiv:2506.18327v1](http://arxiv.org/abs/2506.18327v1)

---

### Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs   and Muslims in Large Language Models: A Systematic Review

**Authors:** Bushra Asseri, Estabrag Abdelaziz, Areej Al-Wabil

**Categories:** cs.CL, cs.AI, cs.CY, cs.HC

**Published:** 2025-06-22T23:15:25Z

**Abstract:** Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility.

**Link:** [arXiv:2506.18199v1](http://arxiv.org/abs/2506.18199v1)

---

### Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to   Detect Amplified and Silenced Perspectives

**Authors:** Batool Haider, Atmika Gorti, Aman Chadha et al.

**Categories:** cs.CL, cs.AI, cs.CY

**Published:** 2025-06-22T18:00:16Z

**Abstract:** Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development.

**Link:** [arXiv:2506.18116v1](http://arxiv.org/abs/2506.18116v1)

---

### The Democratic Paradox in Large Language Models' Underestimation of   Press Freedom

**Authors:** I. Loaiza, R. Vestrelli, A. Fronzetti Colladon et al.

**Categories:** cs.CY, cs.AI, cs.CL, K.4; I.2.7; I.2.0

**Published:** 2025-06-22T14:10:16Z

**Abstract:** As Large Language Models (LLMs) increasingly mediate global information access for millions of users worldwide, their alignment and biases have the potential to shape public understanding and trust in fundamental democratic institutions, such as press freedom. In this study, we uncover three systematic distortions in the way six popular LLMs evaluate press freedom in 180 countries compared to expert assessments of the World Press Freedom Index (WPFI). The six LLMs exhibit a negative misalignment, consistently underestimating press freedom, with individual models rating between 71% to 93% of countries as less free. We also identify a paradoxical pattern we term differential misalignment: LLMs disproportionately underestimate press freedom in countries where it is strongest. Additionally, five of the six LLMs exhibit positive home bias, rating their home countries' press freedoms more favorably than would be expected given their negative misalignment with the human benchmark. In some cases, LLMs rate their home countries between 7% to 260% more positively than expected. If LLMs are set to become the next search engines and some of the most important cultural tools of our time, they must ensure accurate representations of the state of our human and civic rights globally.

**Link:** [arXiv:2506.18045v1](http://arxiv.org/abs/2506.18045v1)

---

### Computational Approaches to Understanding Large Language Model Impact on   Writing and Information Ecosystems

**Authors:** Weixin Liang

**Categories:** cs.CL, cs.AI, cs.CY, cs.HC, cs.LG

**Published:** 2025-06-20T20:15:09Z

**Abstract:** Large language models (LLMs) have shown significant potential to change how we write, communicate, and create, leading to rapid adoption across society. This dissertation examines how individuals and institutions are adapting to and engaging with this emerging technology through three research directions. First, I demonstrate how the institutional adoption of AI detectors introduces systematic biases, particularly disadvantaging writers of non-dominant language varieties, highlighting critical equity concerns in AI governance. Second, I present novel population-level algorithmic approaches that measure the increasing adoption of LLMs across writing domains, revealing consistent patterns of AI-assisted content in academic peer reviews, scientific publications, consumer complaints, corporate communications, job postings, and international organization press releases. Finally, I investigate LLMs' capability to provide feedback on research manuscripts through a large-scale empirical analysis, offering insights into their potential to support researchers who face barriers in accessing timely manuscript feedback, particularly early-career researchers and those from under-resourced settings.

**Link:** [arXiv:2506.17467v1](http://arxiv.org/abs/2506.17467v1)

---

### Are Bias Evaluation Methods Biased ?

**Authors:** Lina Berrayana, Sean Rooney, Luis Garcés-Erice et al.

**Categories:** cs.AI, cs.CL

**Published:** 2025-06-20T16:11:25Z

**Abstract:** The creation of benchmarks to evaluate the safety of Large Language Models is one of the key activities within the trusted AI community. These benchmarks allow models to be compared for different aspects of safety such as toxicity, bias, harmful behavior etc. Independent benchmarks adopt different approaches with distinct data sets and evaluation methods. We investigate how robust such benchmarks are by using different approaches to rank a set of representative models for bias and compare how similar are the overall rankings. We show that different but widely used bias evaluations methods result in disparate model rankings. We conclude with recommendations for the community in the usage of such benchmarks.

**Link:** [arXiv:2506.17111v1](http://arxiv.org/abs/2506.17111v1)

---

### Learning Causally Predictable Outcomes from Psychiatric Longitudinal   Data

**Authors:** Eric V. Strobl

**Categories:** cs.LG, q-bio.QM, stat.ML

**Published:** 2025-06-19T21:56:30Z

**Abstract:** Causal inference in longitudinal biomedical data remains a central challenge, especially in psychiatry, where symptom heterogeneity and latent confounding frequently undermine classical estimators. Most existing methods for treatment effect estimation presuppose a fixed outcome variable and address confounding through observed covariate adjustment. However, the assumption of unconfoundedness may not hold for a fixed outcome in practice. To address this foundational limitation, we directly optimize the outcome definition to maximize causal identifiability. Our DEBIAS (Durable Effects with Backdoor-Invariant Aggregated Symptoms) algorithm learns non-negative, clinically interpretable weights for outcome aggregation, maximizing durable treatment effects and empirically minimizing both observed and latent confounding by leveraging the time-limited direct effects of prior treatments in psychiatric longitudinal data. The algorithm also furnishes an empirically verifiable test for outcome unconfoundedness. DEBIAS consistently outperforms state-of-the-art methods in recovering causal effects for clinically interpretable composite outcomes across comprehensive experiments in depression and schizophrenia.

**Link:** [arXiv:2506.16629v1](http://arxiv.org/abs/2506.16629v1)

---

### Advancing Harmful Content Detection in Organizational Research:   Integrating Large Language Models with Elo Rating System

**Authors:** Mustafa Akben, Aaron Satko

**Categories:** cs.AI, cs.CL

**Published:** 2025-06-19T20:01:12Z

**Abstract:** Large language models (LLMs) offer promising opportunities for organizational research. However, their built-in moderation systems can create problems when researchers try to analyze harmful content, often refusing to follow certain instructions or producing overly cautious responses that undermine validity of the results. This is particularly problematic when analyzing organizational conflicts such as microaggressions or hate speech. This paper introduces an Elo rating-based method that significantly improves LLM performance for harmful content analysis In two datasets, one focused on microaggression detection and the other on hate speech, we find that our method outperforms traditional LLM prompting techniques and conventional machine learning models on key measures such as accuracy, precision, and F1 scores. Advantages include better reliability when analyzing harmful content, fewer false positives, and greater scalability for large-scale datasets. This approach supports organizational applications, including detecting workplace harassment, assessing toxic communication, and fostering safer and more inclusive work environments.

**Link:** [arXiv:2506.16575v1](http://arxiv.org/abs/2506.16575v1)

---

### Automatic Speech Recognition Biases in Newcastle English: an Error   Analysis

**Authors:** Dana Serditova, Kevin Tang, Jochen Steffens

**Categories:** cs.CL, cs.CY, cs.SD, eess.AS

**Published:** 2025-06-19T19:24:12Z

**Abstract:** Automatic Speech Recognition (ASR) systems struggle with regional dialects due to biased training which favours mainstream varieties. While previous research has identified racial, age, and gender biases in ASR, regional bias remains underexamined. This study investigates ASR performance on Newcastle English, a well-documented regional dialect known to be challenging for ASR. A two-stage analysis was conducted: first, a manual error analysis on a subsample identified key phonological, lexical, and morphosyntactic errors behind ASR misrecognitions; second, a case study focused on the systematic analysis of ASR recognition of the regional pronouns ``yous'' and ``wor''. Results show that ASR errors directly correlate with regional dialectal features, while social factors play a lesser role in ASR mismatches. We advocate for greater dialectal diversity in ASR training data and highlight the value of sociolinguistic analysis in diagnosing and addressing regional biases.

**Link:** [arXiv:2506.16558v1](http://arxiv.org/abs/2506.16558v1)

---

### Towards Generalizable Generic Harmful Speech Datasets for Implicit Hate   Speech Detection

**Authors:** Saad Almohaimeed, Saleh Almohaimeed, Damla Turgut et al.

**Categories:** cs.CL, cs.AI, cs.LG

**Published:** 2025-06-19T17:23:08Z

**Abstract:** Implicit hate speech has recently emerged as a critical challenge for social media platforms. While much of the research has traditionally focused on harmful speech in general, the need for generalizable techniques to detect veiled and subtle forms of hate has become increasingly pressing. Based on lexicon analysis, we hypothesize that implicit hate speech is already present in publicly available harmful speech datasets but may not have been explicitly recognized or labeled by annotators. Additionally, crowdsourced datasets are prone to mislabeling due to the complexity of the task and often influenced by annotators' subjective interpretations. In this paper, we propose an approach to address the detection of implicit hate speech and enhance generalizability across diverse datasets by leveraging existing harmful speech datasets. Our method comprises three key components: influential sample identification, reannotation, and augmentation using Llama-3 70B and GPT-4o. Experimental results demonstrate the effectiveness of our approach in improving implicit hate detection, achieving a +12.9-point F1 score improvement compared to the baseline.

**Link:** [arXiv:2506.16476v1](http://arxiv.org/abs/2506.16476v1)

---

### A Hybrid DeBERTa and Gated Broad Learning System for Cyberbullying   Detection in English Text

**Authors:** Devesh Kumar

**Categories:** cs.CL, cs.AI

**Published:** 2025-06-19T06:15:22Z

**Abstract:** The proliferation of online communication platforms has created unprecedented opportunities for global connectivity while simultaneously enabling harmful behaviors such as cyberbullying, which affects approximately 54.4\% of teenagers according to recent research. This paper presents a hybrid architecture that combines the contextual understanding capabilities of transformer-based models with the pattern recognition strengths of broad learning systems for effective cyberbullying detection. This approach integrates a modified DeBERTa model augmented with Squeeze-and-Excitation blocks and sentiment analysis capabilities with a Gated Broad Learning System (GBLS) classifier, creating a synergistic framework that outperforms existing approaches across multiple benchmark datasets. The proposed ModifiedDeBERTa + GBLS model achieved good performance on four English datasets: 79.3\% accuracy on HateXplain, 95.41\% accuracy on SOSNet, 91.37\% accuracy on Mendeley-I, and 94.67\% accuracy on Mendeley-II. Beyond performance gains, the framework incorporates comprehensive explainability mechanisms including token-level attribution analysis, LIME-based local interpretations, and confidence calibration, addressing critical transparency requirements in automated content moderation. Ablation studies confirm the meaningful contribution of each architectural component, while failure case analysis reveals specific challenges in detecting implicit bias and sarcastic content, providing valuable insights for future improvements in cyberbullying detection systems.

**Link:** [arXiv:2506.16052v1](http://arxiv.org/abs/2506.16052v1)

---

### Gender-Neutral Machine Translation Strategies in Practice

**Authors:** Hillary Dawkins, Isar Nejadgholi, Chi-kiu Lo

**Categories:** cs.CL

**Published:** 2025-06-18T17:57:39Z

**Abstract:** Gender-inclusive machine translation (MT) should preserve gender ambiguity in the source to avoid misgendering and representational harms. While gender ambiguity often occurs naturally in notional gender languages such as English, maintaining that gender neutrality in grammatical gender languages is a challenge. Here we assess the sensitivity of 21 MT systems to the need for gender neutrality in response to gender ambiguity in three translation directions of varying difficulty. The specific gender-neutral strategies that are observed in practice are categorized and discussed. Additionally, we examine the effect of binary gender stereotypes on the use of gender-neutral translation. In general, we report a disappointing absence of gender-neutral translations in response to gender ambiguity. However, we observe a small handful of MT systems that switch to gender neutral translation using specific strategies, depending on the target language.

**Link:** [arXiv:2506.15676v1](http://arxiv.org/abs/2506.15676v1)

---

### Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework for   Evaluating Gender Diversity in Large Language Models

**Authors:** Zhengyang Shan, Emily Ruth Diana, Jiawei Zhou

**Categories:** cs.CL

**Published:** 2025-06-18T15:43:16Z

**Abstract:** We present a comprehensive evaluation of gender fairness in large language models (LLMs), focusing on their ability to handle both binary and non-binary genders. While previous studies primarily focus on binary gender distinctions, we introduce the Gender Inclusivity Fairness Index (GIFI), a novel and comprehensive metric that quantifies the diverse gender inclusivity of LLMs. GIFI consists of a wide range of evaluations at different levels, from simply probing the model with respect to provided gender pronouns to testing various aspects of model generation and cognitive behaviors under different gender assumptions, revealing biases associated with varying gender identifiers. We conduct extensive evaluations with GIFI on 22 prominent open-source and proprietary LLMs of varying sizes and capabilities, discovering significant variations in LLMs' gender inclusivity. Our study highlights the importance of improving LLMs' inclusivity, providing a critical benchmark for future advancements in gender fairness in generative models.

**Link:** [arXiv:2506.15568v1](http://arxiv.org/abs/2506.15568v1)

---

### DeVisE: Behavioral Testing of Medical Large Language Models

**Authors:** Camila Zurdo Tagliabue, Heloisa Oss Boll, Aykut Erdem et al.

**Categories:** cs.CL

**Published:** 2025-06-18T10:42:22Z

**Abstract:** Large language models (LLMs) are increasingly used in clinical decision support, yet current evaluation methods often fail to distinguish genuine medical reasoning from superficial patterns. We introduce DeVisE (Demographics and Vital signs Evaluation), a behavioral testing framework for probing fine-grained clinical understanding. We construct a dataset of ICU discharge notes from MIMIC-IV, generating both raw (real-world) and template-based (synthetic) versions with controlled single-variable counterfactuals targeting demographic (age, gender, ethnicity) and vital sign attributes. We evaluate five LLMs spanning general-purpose and medically fine-tuned variants, under both zero-shot and fine-tuned settings. We assess model behavior via (1) input-level sensitivity - how counterfactuals alter the likelihood of a note; and (2) downstream reasoning - how they affect predicted hospital length-of-stay. Our results show that zero-shot models exhibit more coherent counterfactual reasoning patterns, while fine-tuned models tend to be more stable yet less responsive to clinically meaningful changes. Notably, demographic factors subtly but consistently influence outputs, emphasizing the importance of fairness-aware evaluation. This work highlights the utility of behavioral testing in exposing the reasoning strategies of clinical LLMs and informing the design of safer, more transparent medical AI systems.

**Link:** [arXiv:2506.15339v1](http://arxiv.org/abs/2506.15339v1)

---

### Hypothesis Testing for Quantifying LLM-Human Misalignment in Multiple   Choice Settings

**Authors:** Harbin Hong, Sebastian Caldas, Liu Leqi

**Categories:** cs.CY, cs.CL, cs.LG

**Published:** 2025-06-17T22:04:55Z

**Abstract:** As Large Language Models (LLMs) increasingly appear in social science research (e.g., economics and marketing), it becomes crucial to assess how well these models replicate human behavior. In this work, using hypothesis testing, we present a quantitative framework to assess the misalignment between LLM-simulated and actual human behaviors in multiple-choice survey settings. This framework allows us to determine in a principled way whether a specific language model can effectively simulate human opinions, decision-making, and general behaviors represented through multiple-choice options. We applied this framework to a popular language model for simulating people's opinions in various public surveys and found that this model is ill-suited for simulating the tested sub-populations (e.g., across different races, ages, and incomes) for contentious questions. This raises questions about the alignment of this language model with the tested populations, highlighting the need for new practices in using LLMs for social science studies beyond naive simulations of human subjects.

**Link:** [arXiv:2506.14997v1](http://arxiv.org/abs/2506.14997v1)

---

### Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to   Mimic Polarized Social Media Comments

**Authors:** . Pazzaglia, V. Vendetti, L. D. Comencini et al.

**Categories:** cs.CL, cs.CY

**Published:** 2025-06-17T15:41:26Z

**Abstract:** The increasing sophistication of large language models (LLMs) has sparked growing concerns regarding their potential role in exacerbating ideological polarization through the automated generation of persuasive and biased content. This study explores the extent to which fine-tuned LLMs can replicate and amplify polarizing discourse within online environments. Using a curated dataset of politically charged discussions extracted from Reddit, we fine-tune an open-source LLM to produce context-aware and ideologically aligned responses. The model's outputs are evaluated through linguistic analysis, sentiment scoring, and human annotation, with particular attention to credibility and rhetorical alignment with the original discourse. The results indicate that, when trained on partisan data, LLMs are capable of producing highly plausible and provocative comments, often indistinguishable from those written by humans. These findings raise significant ethical questions about the use of AI in political discourse, disinformation, and manipulation campaigns. The paper concludes with a discussion of the broader implications for AI governance, platform regulation, and the development of detection tools to mitigate adversarial fine-tuning risks.

**Link:** [arXiv:2506.14645v1](http://arxiv.org/abs/2506.14645v1)

---

### Digital Gatekeepers: Google's Role in Curating Hashtags and Subreddits

**Authors:** Amrit Poudel, Yifan Ding, Jurgen Pfeffer et al.

**Categories:** cs.CL

**Published:** 2025-06-17T10:10:39Z

**Abstract:** Search engines play a crucial role as digital gatekeepers, shaping the visibility of Web and social media content through algorithmic curation. This study investigates how search engines like Google selectively promotes or suppresses certain hashtags and subreddits, impacting the information users encounter. By comparing search engine results with nonsampled data from Reddit and Twitter/X, we reveal systematic biases in content visibility. Google's algorithms tend to suppress subreddits and hashtags related to sexually explicit material, conspiracy theories, advertisements, and cryptocurrencies, while promoting content associated with higher engagement. These findings suggest that Google's gatekeeping practices influence public discourse by curating the social media narratives available to users.

**Link:** [arXiv:2506.14370v1](http://arxiv.org/abs/2506.14370v1)

---

### hyperFA*IR: A hypergeometric approach to fair rankings with finite   candidate pool

**Authors:** Mauritz N. Cartier van Dissel, Samuel Martin-Gutierrez, Lisette Espín-Noboa et al.

**Categories:** cs.CY, cs.IR, stat.AP

**Published:** 2025-06-17T09:45:08Z

**Abstract:** Ranking algorithms play a pivotal role in decision-making processes across diverse domains, from search engines to job applications. When rankings directly impact individuals, ensuring fairness becomes essential, particularly for groups that are marginalised or misrepresented in the data. Most of the existing group fairness frameworks often rely on ensuring proportional representation of protected groups. However, these approaches face limitations in accounting for the stochastic nature of ranking processes or the finite size of candidate pools. To this end, we present hyperFA*IR, a framework for assessing and enforcing fairness in rankings drawn from a finite set of candidates. It relies on a generative process based on the hypergeometric distribution, which models real-world scenarios by sampling without replacement from fixed group sizes. This approach improves fairness assessment when top-$k$ selections are large relative to the pool or when protected groups are small. We compare our approach to the widely used binomial model, which treats each draw as independent with fixed probability, and demonstrate$-$both analytically and empirically$-$that our method more accurately reproduces the statistical properties of sampling from a finite population. To operationalise this framework, we propose a Monte Carlo-based algorithm that efficiently detects unfair rankings by avoiding computationally expensive parameter tuning. Finally, we adapt our generative approach to define affirmative action policies by introducing weights into the sampling process.

**Link:** [arXiv:2506.14349v1](http://arxiv.org/abs/2506.14349v1)

---

### Meta Optimality for Demographic Parity Constrained Regression via   Post-Processing

**Authors:** Kazuto Fukuchi

**Categories:** stat.ML, cs.LG

**Published:** 2025-06-16T19:36:56Z

**Abstract:** We address the regression problem under the constraint of demographic parity, a commonly used fairness definition. Recent studies have revealed fair minimax optimal regression algorithms, the most accurate algorithms that adhere to the fairness constraint. However, these analyses are tightly coupled with specific data generation models. In this paper, we provide meta-theorems that can be applied to various situations to validate the fair minimax optimality of the corresponding regression algorithms. Furthermore, we demonstrate that fair minimax optimal regression can be achieved through post-processing methods, allowing researchers and practitioners to focus on improving conventional regression techniques, which can then be efficiently adapted for fair regression.

**Link:** [arXiv:2506.13947v1](http://arxiv.org/abs/2506.13947v1)

---

### Addressing Correlated Latent Exogenous Variables in Debiased Recommender   Systems

**Authors:** Shuqiang Zhang, Yuchao Zhang, Jinkun Chen et al.

**Categories:** cs.LG, cs.IR

**Published:** 2025-06-09T07:50:21Z

**Abstract:** Recommendation systems (RS) aim to provide personalized content, but they face a challenge in unbiased learning due to selection bias, where users only interact with items they prefer. This bias leads to a distorted representation of user preferences, which hinders the accuracy and fairness of recommendations. To address the issue, various methods such as error imputation based, inverse propensity scoring, and doubly robust techniques have been developed. Despite the progress, from the structural causal model perspective, previous debiasing methods in RS assume the independence of the exogenous variables. In this paper, we release this assumption and propose a learning algorithm based on likelihood maximization to learn a prediction model. We first discuss the correlation and difference between unmeasured confounding and our scenario, then we propose a unified method that effectively handles latent exogenous variables. Specifically, our method models the data generation process with latent exogenous variables under mild normality assumptions. We then develop a Monte Carlo algorithm to numerically estimate the likelihood function. Extensive experiments on synthetic datasets and three real-world datasets demonstrate the effectiveness of our proposed method. The code is at https://github.com/WallaceSUI/kdd25-background-variable.

**Link:** [arXiv:2506.07517v1](http://arxiv.org/abs/2506.07517v1)

---

### Recommender systems, stigmergy, and the tyranny of popularity

**Authors:** Zackary Okun Dunivin, Paul E. Smaldino

**Categories:** cs.CY, cs.AI, cs.HC, cs.IR

**Published:** 2025-06-06T15:27:23Z

**Abstract:** Scientific recommender systems, such as Google Scholar and Web of Science, are essential tools for discovery. Search algorithms that power work through stigmergy, a collective intelligence mechanism that surfaces useful paths through repeated engagement. While generally effective, this ``rich-get-richer'' dynamic results in a small number of high-profile papers that dominate visibility. This essay argues argue that these algorithm over-reliance on popularity fosters intellectual homogeneity and exacerbates structural inequities, stifling innovative and diverse perspectives critical for scientific progress. We propose an overhaul of search platforms to incorporate user-specific calibration, allowing researchers to manually adjust the weights of factors like popularity, recency, and relevance. We also advise platform developers on how word embeddings and LLMs could be implemented in ways that increase user autonomy. While our suggestions are particularly pertinent to aligning recommender systems with scientific values, these ideas are broadly applicable to information access systems in general. Designing platforms that increase user autonomy is an important step toward more robust and dynamic information

**Link:** [arXiv:2506.06162v1](http://arxiv.org/abs/2506.06162v1)

---

### Quantifying Query Fairness Under Unawareness

**Authors:** Thomas Jaenich, Alejandro Moreo, Alessandro Fabris et al.

**Categories:** cs.IR

**Published:** 2025-06-04T16:31:44Z

**Abstract:** Traditional ranking algorithms are designed to retrieve the most relevant items for a user's query, but they often inherit biases from data that can unfairly disadvantage vulnerable groups. Fairness in information access systems (IAS) is typically assessed by comparing the distribution of groups in a ranking to a target distribution, such as the overall group distribution in the dataset. These fairness metrics depend on knowing the true group labels for each item. However, when groups are defined by demographic or sensitive attributes, these labels are often unknown, leading to a setting known as "fairness under unawareness". To address this, group membership can be inferred using machine-learned classifiers, and group prevalence is estimated by counting the predicted labels. Unfortunately, such an estimation is known to be unreliable under dataset shift, compromising the accuracy of fairness evaluations. In this paper, we introduce a robust fairness estimator based on quantification that effectively handles multiple sensitive attributes beyond binary classifications. Our method outperforms existing baselines across various sensitive attributes and, to the best of our knowledge, is the first to establish a reliable protocol for measuring fairness under unawareness across multiple queries and groups.

**Link:** [arXiv:2506.04140v1](http://arxiv.org/abs/2506.04140v1)

---

### Structured Semantics from Unstructured Notes: Language Model Approaches   to EHR-Based Decision Support

**Authors:** Wu Hao Ran, Xi Xi, Furong Li et al.

**Categories:** cs.IR, cs.AI

**Published:** 2025-06-01T05:07:51Z

**Abstract:** The advent of large language models (LLMs) has opened new avenues for analyzing complex, unstructured data, particularly within the medical domain. Electronic Health Records (EHRs) contain a wealth of information in various formats, including free text clinical notes, structured lab results, and diagnostic codes. This paper explores the application of advanced language models to leverage these diverse data sources for improved clinical decision support. We will discuss how text-based features, often overlooked in traditional high dimensional EHR analysis, can provide semantically rich representations and aid in harmonizing data across different institutions. Furthermore, we delve into the challenges and opportunities of incorporating medical codes and ensuring the generalizability and fairness of AI models in healthcare.

**Link:** [arXiv:2506.06340v1](http://arxiv.org/abs/2506.06340v1)

---

### Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations

**Authors:** Daniele Barolo, Chiara Valentin, Fariba Karimi et al.

**Categories:** cs.CY, cs.AI, cs.DL, cs.IR, cs.SI, physics.soc-ph, 68T50, I.2.7; C.4; F.2; K.4.1

**Published:** 2025-05-29T20:11:11Z

**Abstract:** This paper evaluates the performance of six open-weight LLMs (llama3-8b, llama3.1-8b, gemma2-9b, mixtral-8x7b, llama3-70b, llama3.1-70b) in recommending experts in physics across five tasks: top-k experts by field, influential scientists by discipline, epoch, seniority, and scholar counterparts. The evaluation examines consistency, factuality, and biases related to gender, ethnicity, academic popularity, and scholar similarity. Using ground-truth data from the American Physical Society and OpenAlex, we establish scholarly benchmarks by comparing model outputs to real-world academic records. Our analysis reveals inconsistencies and biases across all models. mixtral-8x7b produces the most stable outputs, while llama3.1-70b shows the highest variability. Many models exhibit duplication, and some, particularly gemma2-9b and llama3.1-8b, struggle with formatting errors. LLMs generally recommend real scientists, but accuracy drops in field-, epoch-, and seniority-specific queries, consistently favoring senior scholars. Representation biases persist, replicating gender imbalances (reflecting male predominance), under-representing Asian scientists, and over-representing White scholars. Despite some diversity in institutional and collaboration networks, models favor highly cited and productive scholars, reinforcing the rich-getricher effect while offering limited geographical representation. These findings highlight the need to improve LLMs for more reliable and equitable scholarly recommendations.

**Link:** [arXiv:2506.00074v1](http://arxiv.org/abs/2506.00074v1)

---

### Graph Contrastive Learning for Optimizing Sparse Data in Recommender   Systems with LightGCL

**Authors:** Aravinda Jatavallabha, Prabhanjan Bharadwaj, Ashish Chander

**Categories:** cs.IR, cs.LG

**Published:** 2025-05-28T17:21:41Z

**Abstract:** Graph Neural Networks (GNNs) are powerful tools for recommendation systems, but they often struggle under data sparsity and noise. To address these issues, we implemented LightGCL, a graph contrastive learning model that uses Singular Value Decomposition (SVD) for robust graph augmentation, preserving semantic integrity without relying on stochastic or heuristic perturbations. LightGCL enables structural refinement and captures global collaborative signals, achieving significant gains over state-of-the-art models across benchmark datasets. Our experiments also demonstrate improved fairness and resilience to popularity bias, making it well-suited for real-world recommender systems.

**Link:** [arXiv:2506.00048v1](http://arxiv.org/abs/2506.00048v1)

---

### Improving Recommendation Fairness without Sensitive Attributes Using   Multi-Persona LLMs

**Authors:** Haoran Xin, Ying Sun, Chao Wang et al.

**Categories:** cs.IR

**Published:** 2025-05-26T03:52:41Z

**Abstract:** Despite the success of recommender systems in alleviating information overload, fairness issues have raised concerns in recent years, potentially leading to unequal treatment for certain user groups. While efforts have been made to improve recommendation fairness, they often assume that users' sensitive attributes are available during model training. However, collecting sensitive information can be difficult, especially on platforms that involve no personal information disclosure. Therefore, we aim to improve recommendation fairness without any access to sensitive attributes. However, this is a non-trivial task because uncovering latent sensitive patterns from complicated user behaviors without explicit sensitive attributes can be difficult. Consequently, suboptimal estimates of sensitive distributions can hinder the fairness training process. To address these challenges, leveraging the remarkable reasoning abilities of Large Language Models (LLMs), we propose a novel LLM-enhanced framework for Fair recommendation withOut Sensitive Attributes (LLMFOSA). A Multi-Persona Sensitive Information Inference module employs LLMs with distinct personas that mimic diverse human perceptions to infer and distill sensitive information. Furthermore, a Confusion-Aware Sensitive Representation Learning module incorporates inference results and rationales to develop robust sensitive representations, considering the mislabeling confusion and collective consensus among agents. The model is then optimized by a formulated mutual information objective. Extensive experiments on two public datasets validate the effectiveness of LLMFOSA in improving fairness.

**Link:** [arXiv:2505.19473v1](http://arxiv.org/abs/2505.19473v1)

---

### Modeling Ranking Properties with In-Context Learning

**Authors:** Nilanjan Sinhababu, Andrew Parry, Debasis Ganguly et al.

**Categories:** cs.IR

**Published:** 2025-05-23T10:58:22Z

**Abstract:** While standard IR models are mainly designed to optimize relevance, real-world search often needs to balance additional objectives such as diversity and fairness. These objectives depend on inter-document interactions and are commonly addressed using post-hoc heuristics or supervised learning methods, which require task-specific training for each ranking scenario and dataset. In this work, we propose an in-context learning (ICL) approach that eliminates the need for such training. Instead, our method relies on a small number of example rankings that demonstrate the desired trade-offs between objectives for past queries similar to the current input. We evaluate our approach on four IR test collections to investigate multiple auxiliary objectives: group fairness (TREC Fairness), polarity diversity (Touch\'e), and topical diversity (TREC Deep Learning 2019/2020). We empirically validate that our method enables control over ranking behavior through demonstration engineering, allowing nuanced behavioral adjustments without explicit optimization.

**Link:** [arXiv:2505.17736v1](http://arxiv.org/abs/2505.17736v1)

---

### A Novel Generative Model with Causality Constraint for Mitigating Biases   in Recommender Systems

**Authors:** Jianfeng Deng, Qingfeng Chen, Debo Cheng et al.

**Categories:** cs.IR

**Published:** 2025-05-22T14:09:39Z

**Abstract:** Accurately predicting counterfactual user feedback is essential for building effective recommender systems. However, latent confounding bias can obscure the true causal relationship between user feedback and item exposure, ultimately degrading recommendation performance. Existing causal debiasing approaches often rely on strong assumptions-such as the availability of instrumental variables (IVs) or strong correlations between latent confounders and proxy variables-that are rarely satisfied in real-world scenarios. To address these limitations, we propose a novel generative framework called Latent Causality Constraints for Debiasing representation learning in Recommender Systems (LCDR). Specifically, LCDR leverages an identifiable Variational Autoencoder (iVAE) as a causal constraint to align the latent representations learned by a standard Variational Autoencoder (VAE) through a unified loss function. This alignment allows the model to leverage even weak or noisy proxy variables to recover latent confounders effectively. The resulting representations are then used to improve recommendation performance. Extensive experiments on three real-world datasets demonstrate that LCDR consistently outperforms existing methods in both mitigating bias and improving recommendation accuracy.

**Link:** [arXiv:2505.16708v1](http://arxiv.org/abs/2505.16708v1)

---

### DiCE-Extended: A Robust Approach to Counterfactual Explanations in   Machine Learning

**Authors:** Volkan Bakir, Polat Goktas, Sureyya Akyuz

**Categories:** cs.AI, cs.LG, cs.NE, I.2; K.4; H.4

**Published:** 2025-04-26T21:22:44Z

**Abstract:** Explainable artificial intelligence (XAI) has become increasingly important in decision-critical domains such as healthcare, finance, and law. Counterfactual (CF) explanations, a key approach in XAI, provide users with actionable insights by suggesting minimal modifications to input features that lead to different model outcomes. Despite significant advancements, existing CF generation methods often struggle to balance proximity, diversity, and robustness, limiting their real-world applicability. A widely adopted framework, Diverse Counterfactual Explanations (DiCE), emphasizes diversity but lacks robustness, making CF explanations sensitive to perturbations and domain constraints. To address these challenges, we introduce DiCE-Extended, an enhanced CF explanation framework that integrates multi-objective optimization techniques to improve robustness while maintaining interpretability. Our approach introduces a novel robustness metric based on the Dice-Sorensen coefficient, ensuring stability under small input variations. Additionally, we refine CF generation using weighted loss components (lambda_p, lambda_d, lambda_r) to balance proximity, diversity, and robustness. We empirically validate DiCE-Extended on benchmark datasets (COMPAS, Lending Club, German Credit, Adult Income) across multiple ML backends (Scikit-learn, PyTorch, TensorFlow). Results demonstrate improved CF validity, stability, and alignment with decision boundaries compared to standard DiCE-generated explanations. Our findings highlight the potential of DiCE-Extended in generating more reliable and interpretable CFs for high-stakes applications. Future work will explore adaptive optimization techniques and domain-specific constraints to further enhance CF generation in real-world scenarios.

**Link:** [arXiv:2504.19027v1](http://arxiv.org/abs/2504.19027v1)

---

