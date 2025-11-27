Full length article

![](images/b815673465bce4d65bbd8d16a3995a1cbef9af32f4f840723f0ca2091ee8f02c.jpg)

# Attention-augmented stockwell transform and convolutional neural network framework for electroencephalogram-based multi-class classification of Frontotemporal Dementia

Siwei Xie $^{a,\ast}$ , Li Xiao $^{b}$ , Haitao Huang $^{c}$ , Dayang Chen $^{d}$ , Haiman Guo $^{e}$ , Amar Jain $^{f}$

a Johns Hopkins Bloomberg School of Public Health, Baltimore, MD, USA  
$^{b}$  School of Mathematics and Information, South China Agricultural University, Guangzhou, Guangdong, China  
$^{\mathrm{c}}$  Department of Mathematics, School of Natural Sciences, University of Manchester, Manchester, UK  
d Bartlett School Env, Energy and Resources, University College London, London, UK  
$^{\mathrm{a}}$  Krieger School of Arts and Sciences, Johns Hopkins University, Baltimore, MD, USA  
$^{\mathrm{f}}$  Symbiosis Institute of Technology, Symbiosis International(Deemed University), Pune, Maharashtra, India

# ARTICLE INFO

Keywords:

Frontotemporal Dementia

CNN

Stockwell-CNN

EEG

Attention mechanism

S-transform

# ABSTRACT

Frontotemporal Dementia (FTD) is often underdiagnosed, despite being the second most common cause of dementia in middle-aged individuals, accounting for approximately  $5\%$  to  $10\%$  of cases. To address this issue, we propose a hybrid Stockwell-CNN model for EEG-based cognitive assessment of FTD. Our model integrates the Stockwell Transform (S-Transform) for time-frequency representation, Convolutional Neural Networks (CNN) for feature extraction, and an attention mechanism to enhance feature selection. The Stockwell Transform converts one-dimensional EEG signals into a two-dimensional time-frequency matrix, which serves as input for a three-block CNN architecture. The CNN extracts hierarchical spatial-temporal features, while the attention mechanism assigns weights to the most relevant features, improving classification performance. The model was trained and evaluated on EEG data from the OpenNeuro (ds004504) dataset, which includes 88 subjects (23 FTD patients, 36 Alzheimer's patients, and 29 healthy controls). Experimental results demonstrate that the Stockwell-CNN model achieves an accuracy of  $96.83\%$  in distinguishing between severe dementia, mild dementia, and normal cognitive states, outperforming traditional deep learning models. The integration of deep learning and time-frequency analysis provides a reliable and effective tool for the early detection and classification of FTD, offering potential clinical applications for timely intervention and personalized treatment strategies.

# 1. Introduction

Dementia is a progressive neurological disease that impairs cognitive function and the ability to perform daily tasks independently. It encompasses disorders such as Alzheimer's disease (AD) and frontotemporal dementia (FTD), each affecting distinct brain regions and cognitive abilities. FTD is the second most prevalent dementia subtype among middle-aged individuals, accounting for  $5\% - 10\%$  of all cases. By primarily targeting the frontal and temporal lobes, FTD leads to deficits in executive function, language, and behavior. As cognitive decline advances, patients increasingly rely on caregivers, imposing significant emotional and financial burdens on families and healthcare systems [1].

Currently, no cure exists for dementia, and available treatments focus solely on symptom management despite ongoing research [2].

The disease progresses through irreversible synaptic degeneration and neuroinflammation, particularly in later stages [3]. Early diagnosis is critical to improving patient outcomes, given the high healthcare costs and limited therapeutic options. Timely detection enables early intervention, better symptom management, and caregiver planning. However, conventional clinical methods struggle to accurately differentiate dementia subtypes, driving interest in reliable biomarkers—particularly those derived from neurophysiological signals like electroencephalography (EEG) [4].

EEG has gained traction as a non-invasive tool for assessing cognitive function in dementia. It captures brain activity and reveals characteristic changes linked to cognitive decline [5]. For instance, resting-state EEG patterns correlate with disease severity: increased theta power and reduced beta power are associated with cognitive

deterioration in mild cognitive impairment (MCI) and AD. However, the relationship between EEG patterns and FTD remains understudied. Identifying EEG biomarkers specific to FTD could enhance early diagnosis and differentiation from other subtypes [6,7].

Neuroimaging techniques such as MRI, PET, and CT provide structural and functional insights into neurodegeneration, detecting atrophy, amyloid plaques, and metabolic dysfunction specific to dementia subtypes. Additionally, cerebrospinal fluid (CSF) and blood biomarkers reveal biochemical signs of neuronal damage and protein misfolding in AD and other dementias. While valuable, these methods are often costly, invasive, or require specialized equipment. In contrast, EEG offers a cost-effective, non-invasive alternative with high temporal resolution for real-time brain activity monitoring. Its dynamic insights into neural oscillations and cognitive impairment make it particularly suited for early detection. Advanced signal processing techniques, such as the Stockwell Transform and convolutional neural networks (CNNs), further improve EEG-based detection accuracy through feature extraction and classification. This positions EEG as a promising tool for enhancing early diagnosis and monitoring, particularly in FTD. In addition to neuroimaging techniques such as MRI, CT, and PET, bio-signaling and emotional signaling modalities have emerged as valuable tools for mental health assessment. Bio-signals—including electrocardiograms (ECG), galvanic skin response (GSR), and electromyograms (EMG)—capture physiological changes related to emotional and cognitive states. For instance, heart rate variability (HRV) derived from ECG has been linked to anxiety, depression, and cognitive workload. Emotional signaling, such as facial expression analysis, vocal prosody, and gaze tracking, provides non-invasive indicators of affective states and social cognition, which are often impaired in dementia patients. These modalities complement EEG by offering multimodal insights into mental health and neurological dysfunction. Integrating such diverse signals in future research may enhance diagnostic robustness, personalize interventions, and deepen understanding of neurodegenerative disease progression.

This study proposes a novel hybrid approach, the Stockwell-CNN model, for EEG-based cognitive assessment in FTD. Unlike prior methods reliant on manual feature extraction or basic machine learning, the Stockwell Transform generates comprehensive time-frequency representations of EEG signals, while CNNs autonomously learn spatiotemporal patterns. An integrated attention mechanism optimizes feature selection by prioritizing relevant EEG patterns, reducing noise, and boosting classification accuracy. Traditional models, such as basic CNNs or feature-based classifiers, often overlook subtle cognitive deficits due to their inability to focus on critical EEG regions. By combining multidimensional feature extraction, deep learning, and attention-driven refinement, the Stockwell-CNN model outperforms these conventional approaches.

# 1.1. Motivation and contribution

1.1 Motivation of the Research Frontotemporal Dementia (FTD), the second most prevalent form of dementia in middle-aged individuals, significantly impairs executive functions, language, and behavior. Accurate and timely diagnosis is critical for effective symptom management and caregiver planning. However, conventional clinical methods struggle to reliably differentiate FTD from other dementia subtypes. While electroencephalography (EEG) is a promising non-invasive diagnostic tool, existing EEG-based approaches often fail to capture the nuanced neural dynamics specific to FTD. To address this challenge, this study proposes a hybrid Stockwell-CNN model that integrates advanced time-frequency analysis (Stockwell Transform), convolutional neural networks (CNNs), and attention mechanisms. This framework enhances the accuracy and interpretability of EEG-based cognitive assessments, bridging a critical gap in the early and precise diagnosis of FTD.

# Key Contributions of the Study:

- Development of a Hybrid Stockwell-CNN Framework: Enhances electroencephalogram (EEG)-based cognitive assessment accuracy in patients with Frontotemporal Dementia (FTD).  
- Integration of Time-Frequency Analysis and Deep Learning: Combines the Stockwell Transform for detailed EEG signal representation with Convolutional Neural Networks (CNNs) for automated feature extraction and classification, improving the detection of cognitive impairments.  
- Interdisciplinary Methodology: Merges signal processing and deep learning to establish a robust framework for early diagnosis and longitudinal monitoring of FTD.  
- Advancement in Neurodegenerative Disorder Research: Addresses the challenge of capturing FTD-specific brain dynamics, offering insights into the complex pathophysiology of the disorder.

Section 2 reviews existing literature, Section 3 details the proposed methodology, and Sections 4, 5, and 6 present the experimental results, discussion, and ablation studies, respectively. Section 7 concludes the study and suggests future research directions.

# 2. Related work

Using electroencephalogram (EEG) data, Ma et al. [8] reported that the accuracy of distinguishing frontotemporal dementia (FTD) from Alzheimer's disease (AD) was lower than that of differentiating healthy elderly controls (HCs) from those with either condition. To address this issue, their study analyzed electrode-to-electrode communication in EEG recordings and incorporated demographic data as features to train machine learning models, specifically support vector machines (SVMs). The primary objective was to enhance the feature set for AD-FTD classification. The initial classification accuracies were  $91.5\%$  for AD-FTD,  $90.4\%$  for FTD-HC, and  $76.9\%$  for AD-HC. Interestingly, feature importance analysis revealed that the features significant for AD-HC classification were not necessarily useful for distinguishing AD from FTD. After removing these redundant features, the classification accuracy for AD-FTD improved to 96.6

To overcome similar challenges, Ou et al. [9] proposed a novel method combining Deep Ensemble Learning (DEL) with two-dimensional convolutional neural networks (2D-CNNs). Their approach integrated advanced supervised deep learning techniques within an ensemble architecture to precisely classify and diagnose EEG data from AD and HC subjects. After preprocessing for noise and artifact reduction, the DEL model was applied to publicly available EEG-based Alzheimer's datasets without explicit feature extraction. The proposed DEL model included five distinct 2D-CNN models as internal classifiers. Consequently, the EEG-based DEL approach achieved strong performance, obtaining an average classification accuracy of  $97.9\%$  using five-fold cross-validation.

Rostamikia et al. [10] developed an EEG-based classification method for distinguishing between AD and FTD. EEG data were collected from 29 HCs, 23 FTD patients, and 36 AD patients. The most discriminative features were selected using Mann-Whitney U-tests and t-tests. Notably, significant differences were observed in the Fp1 channel among FTD and AD patients. Furthermore, alpha and delta subband connectivity measures revealed promising discriminatory potential between these groups. Central brain regions, including the Cz and Pz channels, were found to be critical for diagnosing dementia (AD + FTD vs. HC). Four machine learning (ML) techniques were employed for classification.

Jiang et al. [11] proposed a method leveraging brain functional connectivity features extracted from resting-state EEG signals for diagnosing AD and FTD. They developed a convolutional neural network (CNN) model called Coherence-CNN. EEG recordings (eyes-closed, resting-state) from 29 cognitively normal (CN) individuals, 23 FTD patients,

and 36 AD patients were obtained from publicly available datasets. Spectral clustering was used to identify connectivity patterns associated with disease states. The results showed that while both AD and FTD groups exhibited reduced connectivity, each displayed distinct connectivity profiles, whereas the CN group demonstrated stronger functional connectivity.

Jiao et al. [1] aimed to identify key EEG biomarkers for tracking the progression of AD and distinguishing early-stage patients. Their study included 890 participants: 246 HCs, 189 patients with mild cognitive impairment (MCI), 330 AD patients, and 125 patients with other dementias (e.g., FTD, dementia with Lewy bodies, and vascular cognitive impairment). EEG biomarkers were extracted from resting-state EEG data and used to classify participants into three categories: AD, MCI, and HC. After evaluating classification performance, the most effective biomarkers were identified. Random forest regression models were trained using EEG biomarkers combined with demographic variables (e.g., age and sex), cerebrospinal fluid (CSF) biomarkers, and APOE genotype. The identified EEG biomarkers achieved over  $70\%$  accuracy in three-class classification (AD, MCI, HC).

Miltiadous et al. [12] evaluated six supervised machine learning techniques to classify EEG data from AD and FTD patients. They compared various validation methods, including K-fold cross-validation and leave-one-patient-out cross-validation, to assess model robustness. Their approach demonstrated promising performance, achieving  $86.3\%$  accuracy for FTD detection using random forests and  $78.5\%$  accuracy for AD detection using decision trees.

Si et al. [13] introduced frequency-based multilayer resting-state EEG networks for differentiating AD from FTD. Their findings revealed notable differences in EEG connectivity patterns between the two dementia types. Specifically, AD patients exhibited increased delta-alpha and delta-beta connectivity compared to FTD patients. Moreover, reduced theta-sigma cross-couplings in AD patients, compared to HCs, contributed to the classification results. Using typical network features derived from EEG data, their model achieved an accuracy of  $81.1\%$  for AD vs. FTD classification. Additionally, Mini-Mental State Examination (MMSE) scores were predicted using a multivariable linear regression model based on EEG network topologies.

Wang et al. [14] examined differences in aperiodic EEG activity between AD and FTD, evaluating its utility for differential diagnosis. EEG recordings and cognitive assessments were obtained from 88 participants: 36 AD patients, 23 FTD patients, and 29 CN individuals. By parameterizing neural power spectra, the EEG signal was decomposed to assess group-level differences in various components. A support vector machine was used to assess the diagnostic value of aperiodic features. AD and FTD groups exhibited differences in enhanced alpha power (both raw and periodic) and theta-alpha power ratios compared to the CN group. Additionally, AD patients showed elevated theta power in frontal regions and increased aperiodic parameters (exponent and offset) in the frontal, temporal, central, and parietal regions.

# 2.1. Research gap

The use of Deep Learning (DL) and Machine Learning (ML) methods has made significant strides in the interpretation of biological signals such as EEG and ECG in recent years. However, the complexity of non-stationary signals—especially in neurodegenerative diseases like dementia, particularly Frontotemporal Dementia (FTD)—presents unique challenges for accurate classification. For effective diagnosis of FTD across various stages, capturing dynamic frequency variations over time is critical. Traditional signal analysis techniques, such as the Fourier Transform and Wavelet Transform, often fall short in detecting these subtle temporal changes.

The current limitations in the field include:

(i) Limited utilization of hybrid deep learning models for FTD, and a lack of extensive application of time-frequency analysis techniques in FTD diagnosis.

(ii) Insufficient training data and weak feature extraction methods, leading to overfitting and poor generalization in many EEG-based FTD classification models.  
(iii) Minimal exploitation of multi-channel EEG data, which could otherwise enhance spatial feature representation.  
(iv) The proposed Stockwell-CNN hybrid model, augmented with an attention mechanism, addresses these shortcomings by offering a novel and comprehensive framework for improving cognitive assessment through EEG data.

# 2.2. Limitations of existing models

a. Inadequate feature extraction methods often result in reduced model performance and lower diagnostic accuracy, especially when processing complex signal patterns critical to early-stage FTD detection.  
b. Simplified processing pipelines struggle with high-dimensional EEG data, leading to diminished classification accuracy.  
c. Many EEG-based FTD classification studies suffer from small sample sizes, class imbalance, and inappropriate time-frequency transformation techniques that fail to account for the nuances of EEG data.  
d. There is a lack of comprehensive evaluations comparing various time-frequency transformation techniques—such as the S-Transform, Wavelets, and Short-Time Fourier Transform (STFT)—when combined with different deep learning models for FTD classification.  
e. In most cases, training data are solely used to optimize learning parameters, while validation sets serve to fine-tune models and prevent overfitting. However, insufficient attention is given to robust evaluation using separate test data, which is crucial for assessing the model's generalization capability in detecting cognitive impairments related to FTD.

# 2.3. Problem statement

Frontotemporal Dementia (FTD) is a neurodegenerative disorder that severely impacts cognitive abilities. However, its complex symptoms and overlap with other dementias often lead to underdiagnosis. Current diagnostic methods—such as MRI, PET, and cerebrospinal fluid analysis—are costly, invasive, and hinder early detection and accessibility. While EEG offers a non-invasive, cost-effective alternative for cognitive assessment, existing machine learning systems face challenges in reliable classification due to difficulties in feature extraction and signal interpretation. To address these limitations, this work proposes a hybrid Stockwell-CNN model that integrates the Stockwell Transform for time-frequency representation with CNNs and an attention mechanism for automated feature extraction and selection. The model aims to improve EEG-based cognitive assessments by enabling precise differentiation between severe dementia, moderate dementia, and normal cognitive states. By leveraging deep learning and advanced signal processing, this approach provides a robust, interpretable, and scalable solution for early FTD identification and monitoring, thereby enhancing clinical decision-making and patient outcomes.

# 3. Proposed methodology

The primary goal of the proposed Stockwell-CNN hybrid model with attention is to enhance the accuracy and interpretability of Frontotemporal Dementia (FTD) classification using EEG signals. This approach integrates the S-Transform during preprocessing to ensure precise temporal and spectral localization, capturing comprehensive time-frequency representations of non-stationary EEG data. The model employs three CNN blocks to effectively extract spatial features from these time-frequency representations. Additionally, the attention mechanism works in tandem with fully connected layers to highlight diagnostically relevant features specific to FTD.

![](images/2f7464504d54a069027560fabd4b092a8ebacf0a71e86a2c8c7e31d5d8d2c4a3.jpg)  
Fig. 1. Overview of the proposed Stockwell-CNN model for EEG-based cognitive assessment.

# 3.1. Stockwell-CNN for FTD cognitive assessment

The proposed Stockwell-CNN model is illustrated in Fig. 1 It integrates convolutional neural networks (CNNs) for deep learning, S-Transform for signal processing, and attention mechanisms, creating a framework particularly suited for analyzing time-series data such as EEG signals. This approach aims to address limitations in existing models—including insufficient feature extraction and limited generalizability—to establish a robust foundation for distinguishing between phases of frontotemporal dementia (FTD) with enhanced accuracy. By leveraging the complementary strengths of its components, the hybrid model effectively tackles challenges in anomaly detection and signal classification.

While the inclusion of an attention mechanism prioritizes salient EEG features, it falls short of ensuring full model explainability. To improve transparency, methods such as Gradient-weighted Class Activation Mapping (Grad-CAM) could be integrated. Grad-CAM produces heatmaps that highlight EEG signal regions or time-frequency characteristics critical to classification outcomes. These visualizations offer clinicians insights into brain activity patterns linked to distinct cognitive states (severe, mild, normal). Implementing an explainability framework would not only validate the model's feature selection process but also bolster confidence in its predictions. Such interpretability is essential for clinical adoption, as it provides healthcare professionals with actionable, evidence-based decision-making support.

# 3.1.1. Data collection

Data collection involves acquiring raw EEG signal recordings from sensor-based measurements. In this study, high-quality EEG datasets were obtained from individuals diagnosed with dementia and a demographically matched control group without the condition. The dataset, sourced from OpenNeuro (accession: ds004504), comprises EEG recordings from 88 participants: 23 with frontotemporal dementia (FTD), 36 with Alzheimer's disease, and 29 healthy controls. To facilitate model training, EEG signals were segmented into time-series fragments, ensuring balanced representation across three classes: severe dementia, mild dementia, and normal cognitive function. In addition to electrode frequency data, comprehensive demographic information—including age, gender, and clinical diagnosis—was incorporated to maintain a representative and balanced sample. Furthermore, EEG signals were recorded under standardized conditions to ensure consistency in electrode placement (e.g., adhering to the 10-20 system) and sampling

rates (e.g., 256 Hz), minimizing variability arising from acquisition protocols.

# 3.1.2. Preprocessing

The Savitzky-Golay filter is primarily employed for signal denoising, removing high-frequency noise while preserving signal integrity. For baseline correction, moving average forward filtering techniques are used to eliminate baseline drift and long-term trends. Z-score normalization standardizes signal amplitudes to ensure values fall within a consistent range, facilitating easier analysis. In EEG recordings, Principal Component Analysis (PCA) is often applied to remove non-neural artifacts, such as eye blinks and muscular movements. These preprocessing steps enhance signal quality prior to S-transform analysis. Accurate labeling and annotation of data are critical for training and evaluating machine learning models, particularly in studies involving specific stages of dementia.

# Preprocessing Steps for EEG-Based Analysis:

a. Signal Denoising: Apply the Savitzky-Golay filter to remove high-frequency noise while preserving EEG signal integrity.

b. Baseline Correction: Use moving average forward filtering to eliminate baseline drift and long-term trends.

c. Normalization: Standardize signal amplitudes using Z-score normalization for consistency across samples.

d. Artifact Removal: Implement Principal Component Analysis (PCA) to remove non-neural artifacts (e.g., eye blinks, muscle movements).

e. Segmentation: Divide EEG signals into time-series segments to ensure a balanced dataset for classification.

f. Labeling and Annotation: Accurately label and annotate EEG data for model training and evaluation, ensuring reliable cognitive assessment.

# 3.1.3. Classification by Stockwell-CNN hybrid model

# 3.1.3.1. S-transform (Stockwell transform). :

The novel component in a hybrid model is the S-Transform, also known as the Stockwell Transform. This time-frequency analysis provides enhanced accuracy for analyzing non-stationary signals with changing frequency content over time. The S-Transform builds upon the capabilities of the Continuous Wavelet Transform (CWT) and the Short-Time Fourier Transform (STFT) by offering a localized frequency spectrum over time. This makes it particularly valuable for classifying

![](images/92f58c37e5f7f5839bcc3a737f398e4b762999e53c068f1aa57c1579e8320e5d.jpg)  
Fig. 2. S-Transform pipeline for generating time-frequency features from EEG signals.

![](images/81a45c471ad22f36ec346aec96bf6a200f6b25a94181b526339118a387d2b5ce.jpg)  
Fig. 3. Time-Frequency analysis using CNN-Attention integration.

complex patterns in biomedical signals such as EEG, as it provides precise information about both frequencies and their times of occurrence. By employing a Gaussian window, the S-Transform divides long signals into smaller time segments, enabling localized analysis in both the frequency and time domains. This windowing technique isolates specific signal components for detailed examination. Additionally, resampling methods—primarily using the Fourier Transform—are applied to reduce the sampling rate, effectively lowering computational costs without significantly sacrificing essential signal information. A key advantage of the S-Transform is its ability to handle signals with varying frequencies by combining the strengths of the Fourier and Wavelet transforms. The result is a time-frequency matrix, which is critical for subsequent neural network processing. The S-Transform analysis pipeline is illustrated in Fig. 2

# 3.1.3.2. CNN block:

Three Convolutional Neural Networks (CNN) are proposed in our hybrid model. First, CNN receives the time-frequency matrix generated by the S-Transform. In order to find significant patterns pertaining to various frequencies and time periods, these 3 CNN blocks must extract spatial characteristics from this matrix. The CNN is the best option for tasks requiring the detection of localized patterns in time-frequency data because it can capture hierarchical features and local relationships.

# 3.1.3.3. Attention mechanism:

Following the CNN layers, an Attention Mechanism is utilized to increase the model's performance even more. The model is assisted by the attention mechanism in concentrating on the time-frequency representation's most instructive segments. The attention mechanism makes sure that the model highlights the essential characteristics required for precise classification by giving greater weight to pertinent areas of the signal. For signal processing tasks like EEG classification, where both temporal and spectral information are essential, this S-Transform, CNN, and attention combination creates a very successful model. Time-Frequency Analysis Employing CNN-Attention Integration is displayed in Fig. 3.

# 3.1.4. Cognitive assessment

When handling multi-class problems, categorical cross-entropy is employed for classification jobs. During training, the Adam optimizers are usually used to minimize the loss function. The S-Transform of the signal data, which records both time and frequency information, is utilized to train the model. The model is trained over a variety of epochs appropriate for the amount and complexity of the dataset once it has been split up into batches. The model can be utilized to predict cognitive assessments of dementia into three kinds after training: severe, mild, and normal.

Time-domain wave methods are not as appropriate for precisely identifying problems in EEG signal analysis as the S-transform feature selection is. Evaluating non-stationary data such as EEG depends on a localized time-frequency representation that simultaneously captures both temporal and spectral aspects given by the S-transform. The S-transform derives more significant characteristics than conventional time-domain techniques depending on direct waveform analysis by incorporating changes in frequency across time. It, therefore, improves the sensitivity to minute variations in EEG patterns linked to different neurological diseases or cognitive processes. By using the S-transform, which effectively generates a complete 2D time-frequency matrix from 1D EEG data, CNNs may extract hierarchical features in the Stockwell-CNN hybrid model that enhance classification accuracy. Research results from EEG-based models for cognitive assessment, including the proposed Stockwell-CNN method, show that time-frequency analysis approaches like the S-transform perform noticeably better than pure time-domain analysis in identifying neurodegenerative diseases, including frontotemporal dementia. Consequently, in EEG signal processing, the S-transform provides more diagnostic robustness and precision than time-domain techniques.

# 3.2. Proposed layered architecture

The S-transform process applies the Continuous Wavelet Transform (CWT) on a suitable 1D EEG input. This generates a time-frequency

![](images/d90292beb6020a6813b8171aada531d5016cd205e33d459a1055af033be152c9.jpg)  
Fig. 4. Proposed stockwell-CNN layered architecture.

matrix and serves as an S-transform approximative tool. As the S-transform's output, a 2D matrix that shows the signal's frequency components with time is produced. The Complex Morlet Wavelet (CMW) is employed in our model to analyze time-frequency data. Because it is continuous, CMW is a good fit for CWT. It is composed of a complicated sinusoidal signal modulated by a Gaussian window. The model receives the input shape, which is the time-frequency matrix. Using 32 filters, the first convolutional layer reduces the spatial dimensions while preserving significant characteristics by extracting low-level features (edges, fundamental patterns).

In order to further reduce the spatial dimensions and preserve the most significant information, the second convolutional layer uses 64 filters to extract more complicated patterns. Using 128 filters, the third convolutional layer recovers even more intricate patterns. An attention mechanism then processes the CNN's output, calculating the average value across all spatial dimensions to reduce each feature map to a single scalar. Each feature map's attention weights are calculated by the attention mechanism, and the Softmax activation makes sure that the weights add up to 1. Each feature map is multiplied by its corresponding attention weight once the attention weights have been modified to fit the geometry of the input. A probability (y_pred > 0.75,  $0.5 \leq y\_pred \leq 0.75$ , y_pred < 0.5) representing severe dementia, mild dementia, or normal cognitive state is output by the last layer. Fig. 4. illustrates the suggested Stockwell-CNN Layered architecture.

The Stockwell-CNN layered architecture model typically begins with an input layer that processes modified EEG signals, followed by  $3 \times 3$  convolutional layers that capture fine-grained spatial information. The first convolutional layer may use 32 filters, followed by 64 in subsequent layers to represent EEG patterns hierarchically. Non-linearity and feature learning are enhanced using ReLU activation functions after each convolutional layer. Max-pooling layers with a  $2 \times 2$  window reduce dimensionality while preserving essential information. A fully connected layer classifies the extracted features after the convolutional layers. The final layer employs a Softmax activation function to provide a probability distribution for multi-class classification (severe, mild, normal). Defining these architectural components enhances model transparency and enables researchers to evaluate its performance on diverse datasets.

# 3.3. Derivation for Stockwell-CNN model

The notation for the raw EEG signal is  $x(t)$ , where  $t$  is the time. This shows the signal's amplitude at every time point.

$$
x (t) = \text {E E G S i g n a l} t \tag {1}
$$

The signal in relation to time  $\tau$ , where  $\tau$  is an integration variable in Eq. (1). A window function that fluctuates with frequency  $f$  and revolves around  $t$ . This is usually a Gaussian window in Eq. (2) for the S-transform. The signal is transformed from the time domain to the frequency domain by the Fourier kernel  $e^{-\frac{(\tau - t)^2f^2}{2}}$  in Eq. (2). In Eq. (3), multiply the signal by the window function. To get the frequency components in Eq. (4), apply the Fourier transform to this windowed signal. As a result, a time-frequency matrix  $S(f,t)$  is produced, where each point in Eq. (5) represents the amplitude of a particular frequency  $f$  at time  $t$ .

$$
h (\tau - t, f) = e ^ {- \frac {(\tau - t) ^ {2} f ^ {2}}{2}} \tag {2}
$$

$$
x (\tau) \cdot h (\tau - t, f) \tag {3}
$$

$$
\int_ {- \infty} ^ {\infty} x (\tau) h (\tau - t, f) e ^ {- j 2 \pi f \tau} d \tau \tag {4}
$$

$$
S (f, t) = \int_ {- \infty} ^ {\infty} x (\tau) h (\tau - t, f) e ^ {- j 2 \pi f \tau} d \tau \tag {5}
$$

The S-transform yields the time-frequency matrix  $S(f,t)$  given in Eq. (5). To extract features, this matrix is run through a number of convolutional layers. The convolution operation at layer  $l$  for a 2D input (such as the time-frequency matrix) is given in Eq. (6), where the convolutional weights (filters) are denoted by  $W^{l}$ . Layer  $l$  receives  $Z^{(l-1)}$  as input (in the first layer, this is  $Z^0 = S(f,t)$ ). The convolution operation is shown by  $*$ . The bias term is  $b^{l}$ . The activation function is represented as  $\sigma$  (ReLU).

$$
Z ^ {l} = \sigma \left(W ^ {l} * Z ^ {(l - 1)} + b ^ {l}\right) \tag {6}
$$

Max pooling is used to minimize the feature map's size following each convolution as shown in Eq. (7). Max pooling reduces the spatial

dimensions while preserving the most significant features by choosing the maximum value within a region.

$$
Z _ {\text {p o o l e d}} ^ {I} = \operatorname {P o o l i n g} \left(Z ^ {I}\right) \tag {7}
$$

By giving each section of the feature map a weight, the attention mechanism enables the model to concentrate on the areas that are most pertinent. The formula for computing the attention weights  $\alpha^l$  is given in Eq. (8), where the weights and bias of the attention layer are represented by  $W_{\mathrm{att}}$  and  $b_{\mathrm{att}}$ , and CNN provides the input feature map, which is called  $Z_{\mathrm{pooled}}^l$ . The attention-weighted feature map  $Z_{\mathrm{att}}^l$  is then created by multiplying the attention weights  $\alpha^l$  element-wise with the feature map  $Z_{\mathrm{pooled}}^l$  given in Eq. (9). The attention mechanism draws emphasis to the most important regions of the feature map by utilizing attention weights to scale those regions.

$$
\alpha^ {l} = \operatorname {S o f t m a x} \left(W _ {\text {a t t}} \cdot Z _ {\text {p o o l e d}} ^ {l} + b _ {\text {a t t}}\right) \tag {8}
$$

$$
Z _ {\mathrm {a t t}} ^ {l} = \alpha^ {l} \cdot Z _ {\text {p o o l e d}} ^ {l} \tag {9}
$$

To be fed into the fully linked layers, the attention-weighted feature map  $Z_{\mathrm{att}}^{l}$  is flattened into a 1D vector as shown in Eq. (10). The dense layer applies a non-linearity (ReLU) after a linear transformation by using Eq. (11), where the ReLU activation function is represented by  $\sigma$ , and the weights and biases of the dense layer are represented by  $W_{\mathrm{dense}}$  and  $b_{\mathrm{dense}}$ , respectively. The output layer uses a sigmoid activation function to carry out binary classification for  $y_{\mathrm{pred}}$  and is given in Eq. (12), where the anticipated probability of the positive class, such as dementia, is represented by  $y_{\mathrm{pred}}$ . The output layer's weights and biases are denoted by  $W_{\mathrm{out}}$  and  $b_{\mathrm{out}}$ . The sigmoid function, represented by the symbol  $\sigma$ , converts the output into a probability between 0 and 1.

$$
Z _ {\text {f l a t}} = \text {F l a t t e n} (Z _ {\text {a t t}}) \tag {10}
$$

$$
Z _ {\text {d e n s e}} = \sigma \left(W _ {\text {d e n s e}} \cdot Z _ {\text {f l a t}} + b _ {\text {d e n s e}}\right) \tag {11}
$$

$$
P \left(y _ {i}\right) = \frac {e ^ {z _ {i}}}{\sum_ {j = 1} ^ {K} e ^ {z _ {j}}} \tag {12}
$$

where  $P(y_{i})$  is the probability of class  $i$  (e.g., severe, mild, or normal dementia),  $z_{i}$  is the output of the last fully connected layer for class  $i$ , and  $K$  is the total number of classes. The denominator ensures that the sum of all probabilities equals 1.

The binary cross-entropy loss is minimized in order to train the model, given in Eq. (13).

$$
\operatorname {L o s s} = - \left[ y \log \left(y _ {\text {p r e d}}\right) + (1 - y) \log \left(1 - y _ {\text {p r e d}}\right) \right] \tag {13}
$$

where the true label is denoted by  $y$ , and the model's expected probability is denoted by  $y_{\mathrm{pred}}$ . The network's weights and biases are usually updated using the Adam optimizer in accordance with the gradients of the loss function.

A probability  $y_{\mathrm{pred}}$  is produced by the model for the positive class (dementia). The cognitive state is categorized as normal, mild dementia, and severe dementia based on thresholding. Eq. (14) defines the thresholds for the cognitive classification:

$$
C o g n i t i v e \_ C l a s s = \left\{ \begin{array}{l l} \text {S e v e r e d e m e n t i a ,} & \text {i f y _ {p r e d} > 0 . 7 5} \\ \text {M i l d d e m e n t i a ,} & \text {i f 0 . 5 \leq y _ {p r e d} \leq 0 . 7 5} \\ \text {N o r m a l ,} & \text {i f y _ {p r e d} <   0 . 5} \end{array} \right. \tag {14}
$$

According to the predicted output likelihood, which corresponds to various cognitive states, the subject is categorized in this last phase.

# 3.4. Algorithm

As shown in Algorithm 1 using EEG signal data, this algorithm classifies cognitive states by combining sophisticated methods like the

Algorithm 1: S-Transform-based EEG Classification Using CNN with Attention  
Step 1: Load the unprocessed EEG signal data  $x(t)$ $x(t) = \mathrm{EEG}$  Signal at time  $t$  (From Eq. 1)  
Step 2: Preprocess and determine the S-Transform  $S(f,t)$ , a 2D matrix that gives the 1D signal  $x(t)$  a time-frequency representation.  
 $S(f,t) = \int_{-\infty}^{\infty} x(\tau) h(\tau - t, f) e^{-j2\pi f\tau} d\tau$  (From Eq. 5)  
Step 3: Use the S-Transform  $S(f,t)$  time-frequency data to extract spatial characteristics by applying the  $l$ -th convolutional layer of CNN  $Z^{l}$ .  
 $Z^{l} = \sigma(W^{l} * Z^{(l-1)} + b^{l})$  (From Eq. 6)  
Step 4: To draw emphasis to key elements of the feature map, apply an attention block to the output of the final CNN block to obtain the attention feature map  $Z_{\mathrm{att}}^{l}$ .  
 $Z_{\mathrm{att}}^{l} = \alpha^{l} \cdot Z_{\mathrm{pooled}}^{l}$  (From Eq. 9)  
Step 5: Integrate all layers to obtain the predicted probability  $y_{\mathrm{pred}}$ .  
 $y_{\mathrm{pred}} = \sigma(W_{\mathrm{out}} \cdot Z_{\mathrm{dense}} + b_{\mathrm{out}})$  (From Eq. 12)  
Step 6: Train the model and classify the cognitive assessment into a single class Cognitive-Class based on the predicted output probability.  
Cognitive-Class = {Severe dementia, if  $y_{\mathrm{pred}} > 0.75$  Mild dementia, if  $0.5 \leq y_{\mathrm{pred}} \leq 0.75$  Normal, if  $y_{\mathrm{pred}} < 0.5$  (From Eq. 14)

S-Transform, Convolutional Neural Networks (CNN), and an attention mechanism. In order to transform the one-dimensional EEG signal into a two-dimensional time-frequency representation, the S-Transform is first applied once the raw data has been loaded. This transformation is particularly helpful for studying EEG data since it helps capture how various frequency components change over time. Three CNN blocks are subsequently run through the converted data. These blocks are made up of convolutional and max-pooling layers, which learn local patterns and simplify the data in order to extract spatial information from the time-frequency matrix. The attention mechanism is applied to concentrate on the most instructional regions of the CNN's output following feature extraction. crucial parts of the feature map are given bigger weights by the attention block, which helps the model to emphasize crucial signal qualities like dementia-related aberrant brain activity. Afterward being flattened, the attention mechanism's output is transmitted via fully connected layers, which contain a dropout layer to reduce overfitting. Finally, by use of a sigmoid activation function for binary classification, the output layer forecasts whether the individual exhibits normal cognitive ability or dementia. The model learns to classify cognitive states, such as severe dementia, mild dementia, or normal cognitive function, based on the attributes taken from the EEG signals. It is trained using binary cross-entropy loss on labeled EEG data. This method effectively integrates deep learning, signal processing, and attention to improve the precision of cognitive evaluations derived from EEG data.

# 4. Results

# 4.1. Frontotemporal dementia

EEG recordings from 88 subjects — 23 FTD patients, 36 Alzheimer's patients, and 29 healthy controls are in OpenNeuro (ds004504) dataset [12]. Classifying FTD is ideal for deep learning investigations on neurodegenerative illness diagnosis. EEG signals are raw and pre-processed in BIDS format, and artifact, denoising, and other conventional processing methods are used. The worldwide 10–20 method for electrode

Table 1 Typical EEG electrode signal for FTD.  

<table><tr><td>Position of scalp</td><td>Relevance of FTD</td><td>Electrode</td></tr><tr><td>Left frontal</td><td>Connected to speech production and executive function.</td><td>F3</td></tr><tr><td>Right frontal</td><td>Connected to attention and cognitive processes.</td><td>F4</td></tr><tr><td>Left frontal pole</td><td>Tracks activity in the frontal lobe (decision-making, cognitive processes).</td><td>Fp1</td></tr><tr><td>Right frontal pole</td><td>Tracks activity in the right hemisphere&#x27;s frontal lobes.</td><td>Fp2</td></tr><tr><td>Left lateral frontal</td><td>Observes emotional and linguistic reactions (close to Broca&#x27;s region).</td><td>F7</td></tr><tr><td>Right lateral frontal</td><td>Connected to executive function and emotions.</td><td>F8</td></tr></table>

![](images/21495acf7b251d6f6608dbf32a476e1596bf68847b64ae0d3319ddd5c685052e.jpg)

![](images/0750c4b87ed0573b615ceb048582f7564578e4b1d3f597cd927d6a0da90c685a.jpg)

![](images/daa505cebc026077c39b9053f8ca00347f1cd7cf0e491b005c6afa74798d2e16.jpg)

![](images/81da64359aa46bc23135218db12a3cd07ca099347556f48bc18dc03178461aa9.jpg)

![](images/0a0ce596f37aa6f70af906c32a18a4518bb0aad67d2ccf2f5c5fc0a4b18f6c67.jpg)  
Fig. 5. Raw EEG signals captured from six electrodes commonly associated with frontotemporal dementia (FTD): F3, F4, FP1, FP2, F7, and F8. These scalp positions reflect brain regions involved in executive function, speech, emotion, and cognitive processing. The plots display amplitude variations over time for each channel, providing insight into the temporal dynamics of FTD-relevant brain activity.

![](images/d4cda4ad60d448b24c528c7efb265cd79d1b3fe6eefcf37bcce5c8a01620b70d.jpg)

placement is usually followed when monitoring brain activity using particular EEG electrodes and signal channels in frontotemporal dementia (FTD) research [13]. Although EEG is not typically used to diagnose FTD, specific signal patterns and electrode placements can be used to track cognitive impairments or brain activity in patients with dementia, including FTD. The frontal and temporal lobes are the primary areas affected by FTD, hence EEG data from specific electrode placements over these regions would be relevant. Table 1 depicts a typical EEG electrode signal may be pertinent.

# 4.2. Electrodes analysis for FTD

The use of EEG signals from certain electrodes to investigate FTD is the main topic of the section under "Electrodes Analysis for FTD (Frontotemporal Dementia)". The electrode selection is justified because FTD impacts the frontal and temporal regions. The EEG's amplitude, frequency, and patterns are examined in raw signal analysis, which is displayed in Fig. 5 and offers visual representations of signals from both normal and FTD patients. Prior to performing the S-Transform, preprocessing techniques like filtering, baseline correction, and noise reduction are used to enhance the signal quality. The localization of frequency changes across time is subsequently made possible by the S-Transform, which transforms raw signals into a time-frequency matrix shown in Fig. 6. The S-Transform results are displayed in Fig. 6 along with the distinctions between normal and FTD cases. To relate signal fluctuations to FTD symptoms, specific alterations in brainwave patterns—such as alpha or beta bands—are examined.

# 4.3. Stockwell-CNN model timeframe

The Stockwell-CNN Model's execution time, expressed in milliseconds, is broken down in detail in Table 2. Every step, including the

Stockwell Transform, every CNN block, and the end-to-end execution of the finished model, is timed. This breakdown makes it easier to determine how the model's components share the computational load. The table is enhanced by Fig. 7, which shows the time spent on various CNN layers and processes and graphically depicts the execution time for each level in the model. This aids in locating bottlenecks and potential areas for model architecture optimization.

# 4.4. Evaluation metrics

When assessing a model's performance, evaluation measures are important, mainly for tasks like EEG-based dementia classification. Many metrics, such as accuracy, precision, recall, F1-score, and the area under the curve (AUC) for classification tasks, can be used based on EEG data. Eqs. (15) through (18) characterize the major in assessment metrics and their corresponding formulas in the context of multi-class classification of EEG signals. Evaluation metrics are significant for evaluating a model's performance, mostly when it comes to tasks like categorizing dementia based on EEG data. Numerous metrics, such as accuracy, precision, recall, F1-score, and the AUC for cataloging tasks, can be used based on EEG data. Eq. (15) through 18 signify the main assessment metrics of accuracy, precision, recall, F-score and their corresponding formulas in the context of multi-class categorization of EEG signals.

$$
\text {a c c u r a c y} = \frac {T P + T N}{T P + F P + T N + F N} \tag {15}
$$

$$
\text {p r e c i s i o n} = \frac {T P}{T P + F P} \tag {16}
$$

$$
\text {r e c a l l} = \frac {T P}{T P + F N} \tag {17}
$$

$$
F _ {-} \text {s c o r e} = 2 \cdot \frac {\text {(p r e c i s i o n) (r e c a l l)}}{\text {p r e c i s i o n} + \text {r e c a l l}} \tag {18}
$$

![](images/95ff54cf60557202efbe550583957e723a4751c037c29b5ad957bf4b82788cb6.jpg)

![](images/7c20c6ed1554753f22a2603a4bd8dd363d4c83e8b639ac7d1226a1966a4399aa.jpg)

![](images/cc788f29e06166569db8fbfc744d99d0ee26ab2c62352b8920d184c31410a4a2.jpg)  
Fig. 6. Time-frequency representations of raw EEG signals using the S-transform for FTD-related electrodes: F3, F4, FP1, FP2, F7, and F8. The plots illustrate how signal amplitude varies across both time and frequency, highlighting dominant frequency components. These S-transform outputs provide a detailed spectral-temporal profile of each channel, aiding in the extraction of features relevant to frontotemporal dementia (FTD) diagnosis and analysis.

![](images/022cc934b4d5b291e7ec36fafc9a51138f8f2a2ded736f2e46c4a66283f73fc9.jpg)

![](images/7bbdcbd08a843fead9d0a7b792d979f27223486d58a2b26419c0a0f38518dab7.jpg)

![](images/2b162400728c2f32fbed8b23711be8480f66eed5cb4ddcb9bbc0e74ee016cf00.jpg)

Table 2 Millisecondsof execution time for the Stockwell-CNN.  

<table><tr><td>Signal</td><td>Input to S-transform</td><td>1-CNN block</td><td>2-CNN block</td><td>3-CNN block</td><td>Attention mechanism</td></tr><tr><td>Fp1</td><td>0.4182</td><td>0.378</td><td>0.738</td><td>0.6910</td><td>0.1360</td></tr><tr><td>Fp2</td><td>0.3605</td><td>0.368</td><td>0.688</td><td>0.6870</td><td>0.6930</td></tr><tr><td>F3</td><td>0.2403</td><td>0.402</td><td>0.690</td><td>0.6900</td><td>0.6920</td></tr><tr><td>F4</td><td>0.3450</td><td>0.369</td><td>0.417</td><td>0.3360</td><td>0.6970</td></tr><tr><td>F7</td><td>0.4730</td><td>0.246</td><td>0.732</td><td>0.6870</td><td>0.6940</td></tr><tr><td>F8</td><td>0.4870</td><td>0.756</td><td>0.554</td><td>0.3078</td><td>0.4671</td></tr></table>

K-fold cross-validation is a method used for testing the performance of machine learning and deep learning models. In particular, we utilized 10-fold cross-validation, which divides each dataset into 10 groups, of which 9 are used for training, and the remaining set is used for testing. The procedure of training and testing is repeated k times, with k being 10 in our example. A Hybrid Stockwell-CNN Method for EEG-based Analysis in Cognitive evaluation of Frontotemporal Dementia, the introduction of 10-fold cross-validation greatly boosts model robustness and reliability. Cross-validation is a statistical technique in which the dataset is split into ten parts, or subsets, with nine being used for training and one for validation. The proposed model achieved a superior performance of  $96.83\%$  for our input data by applying 10-fold cross-validation. The accuracy performance of the current AlexNet, InceptionV3, ResNet-50, and SequeezeNet, in contrast, was  $94.05\%$ ,  $95.80\%$ ,  $95.80\%$ , and  $86.37\%$ , respectively, as indicated in Tables 3 and 4.

The ROC curve shown in Fig. 8 illustrates the performance of a classification model, with the True Positive Rate (TPR) plotted against the False Positive Rate (FPR). The curve demonstrates strong predictive ability, as indicated by the area under the curve (AUC) value of 0.96. A higher AUC (closer to 1) suggests that the model has excellent discrimination capability between positive and negative classes, making it a highly effective classifier for the given task

Table 5 presents the precision, recall, and F1-score corresponding to different cognitive levels of frontotemporal dementia (FTD), evaluating the model's classification performance for severe dementia, mild dementia, and normal cognitive function.

For severe dementia, the model achieves a high precision of  $97.65\%$ , indicating that nearly all cases predicted as severe are indeed correct. With a recall of  $93.29\%$ , the model successfully identifies the majority

Table 3 10-fold cross validation of proposed model analysis.  

<table><tr><td>K-folds</td><td>Accuracy</td></tr><tr><td>1</td><td>0.96</td></tr><tr><td>2</td><td>0.95</td></tr><tr><td>3</td><td>0.97</td></tr><tr><td>4</td><td>0.97</td></tr><tr><td>5</td><td>0.96</td></tr><tr><td>6</td><td>0.94</td></tr><tr><td>7</td><td>0.96</td></tr><tr><td>8</td><td>0.96</td></tr><tr><td>9</td><td>0.97</td></tr><tr><td>10</td><td>0.95</td></tr><tr><td>10-Fold mean</td><td>0.96</td></tr></table>

Table 4 Comparison of the proposed model with the state-of-the-art methods for validation analysis.  

<table><tr><td>Authors</td><td>Evaluation methods</td><td>Accuracy (%)</td></tr><tr><td>AlexNet</td><td>10-fold cross-validation</td><td>94.05</td></tr><tr><td>InceptionV3</td><td>10-fold cross-validation</td><td>95.80</td></tr><tr><td>ResNet-50</td><td>10-fold cross-validation</td><td>91.11</td></tr><tr><td>SqueezeNet</td><td>10-fold cross-validation</td><td>86.37</td></tr><tr><td>Our model</td><td>10-fold cross-validation</td><td>96.83</td></tr></table>

of actual severe cases, missing only a small fraction. The resulting F1-score of  $95.41\%$  reflects a well-balanced performance, combining both precision and recall effectively in this category.

In the case of mild dementia, the model attains a precision of  $80.24\%$ , which suggests the presence of some false positives. However, the recall of  $95.69\%$  indicates strong sensitivity in detecting true mild

![](images/c8e357ca56fc5f80e768d5edb6cbccd10c7115cb91faf8ab446a507b398eb14b.jpg)  
Fig. 7. Stockwell-CNN pipeline outputs across five processing stages for FTD EEG signals, demonstrating spatial and temporal feature refinement.

![](images/d8ed0ad84af820c27ca95405dfcb260e793adf70fce2e524d7614327839ce4d0.jpg)  
Fig. 8. ROC curve analysis.

dementia cases. The F1-score of  $84.37\%$  shows that, although slightly less consistent than for severe cases, the model still performs robustly in this classification task.

For the normal cognitive group, the model achieves a precision of  $95.21\%$ , demonstrating high confidence in its predictions for normal cases. However, the recall is lower at  $80.07\%$ , suggesting that some

normal cases may be incorrectly classified as dementia. The F1-score of  $86.15\%$  reflects good performance overall, though with room for improvement, particularly in recall.

Fig. 9(a) illustrates the relationship between Precision and Cognitive Level for FTD. The model's precision improves with the severity of cognitive impairment, as depicted in the "Precision vs. Cognitive Level

![](images/0cfe58c8f05dbbfe35512a96fe7296b90ef5cb7c2ea1e49235ae12f6f630d71b.jpg)  
a

![](images/66d9c6e6bd77535d1bdcef64c633ba60f6e68163c457fbd43b1e8bc682cd0cfc.jpg)  
b

![](images/1c93ed290651540f3aa867f14947d2649516afd30c39b8fb5497415d62316fb6.jpg)  
C  
Fig. 9. (a) Precision vs. Cognitive Level for FTD, (b) Recall vs. Cognitive Level for FTD, (c) F1-Score vs. Cognitive Level for FTD.

Table 5 Metrics for cognitive level for FTD.  

<table><tr><td>Cognitive level for FTD</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>Severe dementia</td><td>97.65%</td><td>93.29%</td><td>95.41%</td></tr><tr><td>Mild dementia</td><td>80.24%</td><td>95.69%</td><td>84.37%</td></tr><tr><td>Normal</td><td>95.21%</td><td>80.07%</td><td>86.15%</td></tr></table>

Table 6 Accuracy for cognitive level for FTD.  

<table><tr><td>Cognitive level for FTD</td><td>Time frequency representation</td><td>Accuracy</td></tr><tr><td>GNN</td><td>Wavelet Packet Decomposition (WPD)</td><td>88.56%</td></tr><tr><td>RNN</td><td>Short-Time Fourier Transform (STFT)</td><td>89.05%</td></tr><tr><td>MLP</td><td>Wavelet Transform (WT)</td><td>92.06%</td></tr><tr><td>RBFN</td><td>Fast Fourier Transform (FFT)</td><td>91.25%</td></tr><tr><td>Stockwell-CNN</td><td>S-Transform with attention mechanism</td><td>96.83%</td></tr></table>

for FTD" graph. Precision is approximately  $80.24\%$  for mild dementia, increases to  $95.21\%$  for individuals with normal cognition, and reaches its highest value of 97.65 for severe dementia. This trend suggests that the model becomes more accurate in correctly identifying positive dementia cases as the severity increases. Notably, the precision difference between moderate and normal cognition highlights a significant improvement in model accuracy with advancing impairment.

Fig. 9(b) presents the comparison of Recall across different cognitive levels for FTD, displaying a U-shaped trend in the "Recall vs. Cognitive Level for FTD" graph. The model exhibits high sensitivity for mild dementia with a recall of  $95.69\%$ . However, recall significantly drops to  $80.07\%$  for individuals with normal cognition, indicating a tendency to misclassify some normal cases as dementia. Interestingly, recall rises again to  $93.29\%$  for severe dementia, showing that the model is more effective at capturing true positive cases when dementia is more advanced. This pattern indicates stronger performance in detecting both mild and severe dementia cases compared to normal cognitive function.

Fig. 9(c) displays the F1-score in relation to cognitive levels for FTD. The "F1-score vs. Cognitive Level for FTD" graph shows improved model performance with increasing severity of dementia. The F1-score is  $84.37\%$  for mild dementia,  $86.15\%$  for normal cognition, and peaks at  $95.41\%$  for severe dementia. These results confirm that the model achieves optimal balance between precision and recall in severe cases. Overall, as cognitive impairment deepens, particularly in severe dementia, the model demonstrates enhanced reliability and classification effectiveness, reflecting consistent improvements in both sensitivity and precision.

Fig. 10 illustrates the accuracy comparison with the proposed Stockwell-CNN model, while Table 6 presents a detailed comparison of various deep learning models based on EEG data for classifying cognitive levels in frontotemporal dementia (FTD). The GNN model achieves an accuracy of  $88.56\%$ , reflecting a moderate classification

performance. The RNN, well-suited for sequential data analysis, offers a slight improvement, reaching  $89.05\%$  accuracy. A notable increase is observed with the MLP model, which achieves  $92.06\%$ , indicating that a simple feedforward neural network with multiple layers is effective for this task. The RBFN model, leveraging radial basis functions for classification, performs comparably well with  $91.25\%$  accuracy, though slightly lower than MLP. The highest performance is recorded by the proposed Stockwell-CNN model, which integrates the S-Transform, Convolutional Neural Networks (CNN), and an Attention Mechanism, achieving an outstanding accuracy of  $96.83\%$ . This demonstrates the effectiveness of combining advanced time-frequency representation with deep feature extraction and attention-based refinement for accurate classification of FTD cognitive levels.

# 4.5. Statistical analysis

# 4.5.1. Analyzing fit curves in statistical models for parameter estimation and model validation

In order to visually examine the relationship between independent and dependent variables and spot trends and patterns in your data, you can plot a fit curve in JMP. By showing how well the fitted curve resembles the real data points, it helps you evaluate how good your model is. Moreover, it provides details on parameter estimations and statistical relevance, thereby guiding analysts and researchers in their decisions. Finally, it makes customizing and results exporting for presentations or reports easy. Fig. 11 shows the Fit Curve for Logistic 4P and Gaussian peak (Signals vs. C3).

# 4.5.2. Parameter estimation

The estimation of parameter values for Growth Rate, Inflection Point, Lower Asymptote, and Upper Asymptote is essential for documenting the nonlinear behavior of EEG signals in cognitive assessments, particularly in Frontotemporal Dementia (FTD). Dynamic fluctuations in EEG signal patterns were modeled using the proposed approach during cognitive processing. Growth Rate indicates the fast response speed of the brain, so reflecting the changes in EEG pattern frequency. The Inflection Point in FTD patients is the moment or frequency at which the EEG signal shows a significant change, therefore indicating cognitive deterioration. The minimum and maximum EEG response limits defined by the Lower Asymptote and Upper Asymptote therefore defining the limits of brain activity variations. The hybrid Stockwell-CNN approach effectively collects and classifies EEG signal characteristics, therefore guaranteeing a strong cognitive assessment in FTD. This efficiency is facilitated by the accurate estimation of these parameters. The decision to employ this approach was driven by the need to optimize the deep learning model's performance by capturing signal dynamics that are directly correlated with cognitive impairments.

The estimations from a growth model in Table 7 that are presented in this analysis vary in significance depending on the parameters. The Growth Rate shows a little downward trend  $(-0.006)$  in the data and

![](images/259ff3a1a7190872aade6f98c2fa812912060c1175e157689e9360ecca01fe04.jpg)  
Fig. 10. ROC curve analysis.

![](images/da5d9a7eb76aed3cf41213d3ce0111388d642e30aa33133b2d4ef0ddb3d437c7.jpg)

![](images/180c24999e5e1d3d3f8006f4b4d194d3fc26a1b97271464b0fb6bbf0c941a350.jpg)

![](images/3ada33ad639880ca1aa11d13ed3e37cfa441fa2681572cd5b205d576bd8f2200.jpg)

![](images/a6681cb2ddb4599d85ae39ce1ec6c1017ec998ebbbb4f508e25bf16e18139154.jpg)

![](images/3a1ecef2bf38877703e8428ee3c6db92a2894558265a385ff164ba7ec707782a.jpg)  
Fig. 11. Plot for Fit curve (Signals vs. C3).

![](images/ade64d9260e047cf36968be99661c3d3369a1b981bc83133c24e0b53edcf36d5.jpg)

![](images/71d9c9e470f277fd3b69b3427ac3332fd600e2704764d3b0c41006aa9db2bdfa.jpg)

Table 7 Parameter estimation for Fp1 vs.C3.  

<table><tr><td>Parameter</td><td>Estimate</td><td>Std error</td><td>Wald ChiSquare</td><td>Prob &gt; ChiSquare</td><td>Lower 95%</td><td>Upper 95%</td></tr><tr><td>Growth rate</td><td>-0.006051</td><td>0.0018398</td><td>10.817873</td><td>0.0010</td><td>-0.009657</td><td>-0.002445</td></tr><tr><td>Inflection point</td><td>2358.4078</td><td>10677258</td><td>4.8789e-8</td><td>0.9998</td><td>-20924683</td><td>20929400</td></tr><tr><td>Lower Asymptote</td><td>-14253791</td><td>9.209e+11</td><td>2.396e-10</td><td>1.0000</td><td>-1.8e+12</td><td>1.805e+12</td></tr><tr><td>Upper Asymptote</td><td>9.0394355</td><td>2.975944</td><td>9.2264177</td><td>0.0024</td><td>3.2066925</td><td>14.872179</td></tr></table>

is statistically significant  $(p = 0.001)$ . Additionally important  $(p = 0.0024)$  is the Upper Asymptote, which points to a reasonable upper limit for growth of 9.04. With huge standard errors and broad confidence ranges, the Inflection Point and Lower Asymptote are not significant, showing great uncertainty and imprecision in those predictions. Overall, the model yields trustworthy information about growth rate and upper bounds, but it is equivocal about the other parameters.

None of the factors in this analysis shown in Table 8 are statistically significant, although it does yield estimates from a growth model. There is uncertainty in the Growth Rate (0.002), which is insignificant  $(\mathrm{p} = 0.7912)$  with a broad confidence interval. Because of its large standard error and negligible  $p$  -value (0.9997), the Inflection Point (4799.67) is untrustworthy. Likewise, the Upper Asymptote (2,176,599.9) and Lower Asymptote (-141.51) have wide confidence ranges and large

errors, indicating negligible relevance. Overall, the model's significant imprecision indicates that it is not very reliable for drawing conclusions or making predictions.

Estimates for the logistic model's parameters are shown in Table 9. At the  $5\%$  level, the Growth Rate is 0.1505 (standard error: 0.0694), with a  $95\%$  confidence range between 0.0144 and 0.2865, a Wald Chi-Square of 4.698, and a  $p$ -value of 0.0302. The Inflection Point is 47.95 (standard error: 4.803), with a confidence interval of 38.54 to 57.36 and strong significance ( $p$ -value  $< 0.0001$ ). The Lower Asymptote has a confidence interval of  $-0.1781$  to 0.0558 and is assessed to be  $-0.0611$  (standard error: 0.0597), however it is not statistically significant ( $p$ -value  $= 0.3057$ ). High significance ( $p$ -value  $< 0.0001$ ) is demonstrated by the Upper Asymptote of 5.1464 (standard error: 0.8140), with a

Table 8 Parameter estimation for Fp2 vs.C3.  

<table><tr><td>Parameter</td><td>Estimate</td><td>Std error</td><td>Wald ChiSquare</td><td>Prob &gt; ChiSquare</td><td>Lower 95%</td><td>Upper 95%</td></tr><tr><td>Growth rate</td><td>0.0020088</td><td>0.0075862</td><td>0.0701164</td><td>0.7912</td><td>-0.01286</td><td>0.0168775</td></tr><tr><td>Inflection point</td><td>4799.6667</td><td>13964636</td><td>1.1813e-7</td><td>0.9997</td><td>-27365385</td><td>27374984</td></tr><tr><td>Lower Asymptote</td><td>-141.5134</td><td>276.38796</td><td>0.2621542</td><td>0.6086</td><td>-683.2238</td><td>400.19705</td></tr><tr><td>Upper Asymptote</td><td>2176599.9</td><td>6.098e+10</td><td>1.2739e-9</td><td>1.0000</td><td>-1.2e+11</td><td>1.195e+11</td></tr></table>

Table 9 Parameter estimation summary.  

<table><tr><td>Parameter</td><td>Estimate</td><td>Std error</td><td>Wald ChiSquare</td><td>Prob &gt; ChiSquare</td><td>Lower 95%</td><td>Upper 95%</td></tr><tr><td>Growth rate</td><td>0.1504754</td><td>0.0694243</td><td>4.6979455</td><td>0.0302*</td><td>0.0144063</td><td>0.2865445</td></tr><tr><td>Inflection point</td><td>47.949718</td><td>4.8027591</td><td>99.675979</td><td>&lt;.0001*</td><td>38.536483</td><td>57.362953</td></tr><tr><td>Lower Asymptote</td><td>-0.061109</td><td>0.059666</td><td>1.0489479</td><td>0.3057</td><td>-0.178052</td><td>0.0558344</td></tr><tr><td>Upper Asymptote</td><td>5.1463532</td><td>0.8140192</td><td>39.969603</td><td>&lt;.0001*</td><td>3.5509048</td><td>6.7418016</td></tr></table>

Table 10 Parameter estimation for F4 vs.C3.  

<table><tr><td>Parameter</td><td>Estimate</td><td>Std error</td><td>Wald ChiSquare</td><td>Prob &gt; ChiSquare</td><td>Lower 95%</td><td>Upper 95%</td></tr><tr><td>Growth rate</td><td>0.0018655</td><td>0.000848</td><td>4.8393707</td><td>0.0278*</td><td>0.0002034</td><td>0.0035275</td></tr><tr><td>Inflection point</td><td>5006.7451</td><td>1.242341.5</td><td>1.6242e-5</td><td>0.9968</td><td>-2429938</td><td>2439951.4</td></tr><tr><td>Lower Asymptote</td><td>-412.8188</td><td>115.38657</td><td>12.799954</td><td>0.0003*</td><td>-638.9723</td><td>-186.6653</td></tr><tr><td>Upper Asymptote</td><td>4697526.6</td><td>1.087e+10</td><td>1.8678e-7</td><td>0.9997</td><td>-2.13e+10</td><td>2.131e+10</td></tr></table>

Table 11 Parameter Estimation for F7 vs.C3.  

<table><tr><td>Parameter</td><td>Estimate</td><td>Std error</td><td>Wald ChiSquare</td><td>Prob &gt; ChiSquare</td><td>Lower 95%</td><td>Upper 95%</td></tr><tr><td>Growth rate</td><td>-0.025995</td><td>0.0013163</td><td>389.99809</td><td>&lt;.0001*</td><td>-0.028575</td><td>-0.023415</td></tr><tr><td>Inflection point</td><td>3.3506259</td><td>1.5590722</td><td>4.6186962</td><td>0.0316*</td><td>0.2949004</td><td>6.4063513</td></tr><tr><td>Lower Asymptote</td><td>-95.18372</td><td>4.0688675</td><td>547.24054</td><td>&lt;.0001*</td><td>-103.1586</td><td>-87.20889</td></tr><tr><td>Upper Asymptote</td><td>87.253616</td><td>4.2905514</td><td>413.56202</td><td>&lt;.0001*</td><td>78.84429</td><td>95.662942</td></tr></table>

confidence range ranging from 3.5509 to 6.7418. Overall, the lower asymptote is not important, but the growth rate and asymptotes are.

The logistic model's parameter estimations are shown in Table 10. A statistically significant  $p$ -value of 0.0278 is obtained by estimating the Growth Rate to be 0.00187 with a standard error of 0.00085 and a Wald Chi-Square of 4.839. Given that the  $95\%$  CI ranges from 0.0002 to 0.0035, this suggests that the growth rate is significant. With a wide confidence interval between -2,429,938 and 2,439,951, the Inflection Point estimate of 5006.7451 has a very large standard error and an insignificant  $p$ -value of 0.9968, suggesting that this parameter lacks statistical reliability. With a  $p$ -value of 0.0003 and a standard error of 115.387, the Lower Asymptote is -412.8188, indicating a statistically significant range of -638.9723 to -186.6653. Finally, the calculated Upper Asymptote of 4,697,526.6 is not statistically significant ( $p$ -value = 0.9997) and has a huge confidence interval ranging from -2.13e+10 to 2.131e+10, indicating substantial uncertainty in this estimate.

The parameter estimates for a model are given in Table 11. With a  $95\%$  confidence interval between  $-0.028575$  and  $-0.023415$ , the Growth Rate is predicted to be negative at  $-0.025995$  with a standard error of 0.0013163 and a very significant Wald Chi-Square value of 389.99809 ( $p$ -value  $< 0.0001$ ). The predicted Inflection Point is 3.3506259, with a significant Wald Chi-Square value of 4.6186962 ( $p$ -value  $= 0.0316$ ) and a standard error of 1.5590722. With a range of 0.2949004 to 6.4063513, the confidence interval indicates some variation in the estimate. With a confidence interval ranging from  $-103.1586$  to  $-87.20889$  and a highly significant Wald Chi-Square value of 547.24054 ( $p$ -value  $< 0.0001$ ), the Lower Asymptote is  $-95.18372$ . Along with being highly significant ( $p$ -value  $< 0.0001$ ), the Upper Asymptote is 87.253616, with a Wald Chi-Square value of 413.56202 and a confidence interval ranging from 78.84429 to 95.662942. According to these findings, the asymptotes of the model are well-defined and substantially distinct from zero.

The statistical model results in Table 12 provide important information about the parameters being analyzed. With a significant negative estimate of  $-0.013852$  and a  $p$ -value of less than 0.0001, the growth

rate indicates a decrease as the independent variable increases. There is no compelling evidence of a directional change, as the estimated inflection point is 3.1156538 but is not statistically significant  $(p = 0.1772)$ . The answer approaches this value at infinity, according to the lower asymptote's significant estimate of  $-106.0722$ , which is backed by a  $p$ -value of less than 100001. Similarly, the higher asymptote has a significant positive estimate of 101.55121 and a  $p$ -value of less than 0.0001. In conclusion, the inflection point is not significant, but the growth rate and both asymptotes are statistically significant.

Overall, parameter estimates between FTD signals serve as crucial tools for model validation, enabling researchers to evaluate the effectiveness and reliability of their statistical models.

# 4.5.3. Statistical model comparison for Stockwell-CNN model

Models with more parameters are penalized by the Corrected Akaike Information Criterion (AICc), a model selection criterion that accounts for short sample sizes. An additional model selection tool that is comparable to AICc but has a larger penalty for complexity is the Bayesian Information Criterion, or BIC.

The comparison of statistical models for the Stockwell-CNN model is visualized in Fig. 12. The sum of squared errors, or SSE, records all squared differences between observed and projected values, whereas the mean squared error, or MSE, averages the total squared differences to get the average model error. In order to aid interpretation, the square root of mean square error, or RMSE (Root Mean Squared Error), offers an error measure in units that are compatible with the predicted values. The R-Square (Coefficient of Determination), which displays the proportion of variance in the dependent variable that can be explained by the independent variables, is a measure of fit and is closer to 1.

Six EEG signal channels (Fp1, FP2, F3, F4, F7, and F8) are used in the table to evaluate the performance of two statistical models, Logistic 4P and Gaussian Peak, utilizing assessment metrics such AICc, BIC, SSE, MSE, RMSE, and R-Square. In general, the Logistic 4P model performs better than the Gaussian Peak model in the majority of signal channels. Lower AICc, BIC, SSE, MSE, and RMSE values, which signify a better

Table 12 Parameter estimation for F8 vs.C3.  

<table><tr><td>Parameter</td><td>Estimate</td><td>Std Error</td><td>Wald ChiSquare</td><td>Prob &gt; ChiSquare</td><td>Lower 95%</td><td>Upper 95%</td></tr><tr><td>Growth rate</td><td>-0.013852</td><td>0.0008168</td><td>287.60911</td><td>&lt;.0001*</td><td>-0.015453</td><td>-0.012251</td></tr><tr><td>Inflection point</td><td>3.1156538</td><td>2.3086842</td><td>1.821249</td><td>0.1772</td><td>-1.409284</td><td>7.6405917</td></tr><tr><td>Lower Asymptote</td><td>-106.0722</td><td>5.3959895</td><td>386.42155</td><td>&lt;.0001*</td><td>-116.6482</td><td>-95.4963</td></tr><tr><td>Upper Asymptote</td><td>101.55121</td><td>6.2139195</td><td>267.0785</td><td>&lt;.0001*</td><td>89.372148</td><td>113.73027</td></tr></table>

![](images/fe94f9eea6e075f04f9ce3471dd4eef642a155228d54f9ee1f9e1abf149ccccb.jpg)

![](images/e185d7c119cb3fbf509cfa114a19f001600aab0f8892f18264c88f6d130b75e6.jpg)

![](images/ed46ed32c09fc6143d939e5a420d08c6c402197d644421335088c402ac709cd1.jpg)

![](images/61bb9d3e0295358702493de9e72c6564026037fe5dfd8061201081c18fc3a5a1.jpg)  
Fig. 12. Comparison of model performance metrics between Logistic 4-Parameter (4P) and Gaussian Peak models for various FTD EEG signals (Fp1-F8). The metrics evaluated include (a) AICc, (b) BIC, (c) RMSE, (d) SSE, and (e) R-Square. Across multiple FTD signal channels, the Logistic 4P model demonstrates superior performance in terms of lower RMSE and SSE and higher R-Square values, indicating a better overall fit compared to the Gaussian Peak model.

![](images/288297f7da6c9f9ef671da99279e6b4614ea35cad60769e8273e54ea4acebb65.jpg)

fit and fewer prediction errors, make this clear. Additionally, the R-Square values demonstrate the higher explanatory power of Logistic 4P, especially for signals such as F4, F7, and F8, where it accounts for a substantial amount of variation (as high as 0.81 for F8). When it comes to F4 and F7, where the error metrics differ significantly, the Gaussian Peak model performs worse. Even in channels like Fp1 and F3 where the models perform comparably, Logistic 4P maintains a tiny edge. For identifying the patterns in these EEG data, the Logistic 4P model turns out to be a more trustworthy choice overall. The Stockwell-CNN Model Statistical Model Comparison is displayed in Table 13.

In summary, the Stockwell-CNN model is likely to perform better than conventional statistical models in applications involving complex time-frequency data for Parameter estimation, analyzing fit curves and

model comparison, but it requires more processing power and has limited interpretability.

# 5. Ablation study

# 5.1. Inference on using attention mechanism and S-transform

The proposed Stockwell-CNN method is especially useful in cognitive assessments, as EEG signals exhibit high-dimensional and nonstationary properties that are difficult to capture using traditional approaches. The CNN effectively detects subtle patterns and abnormalities in brain activity associated with FTD, leading to excellent discriminatory power between healthy individuals and those affected

Table 13 Statistical model comparison for Stockwell-CNN Model.  

<table><tr><td>FTD signals</td><td>Model</td><td>AICc</td><td>AICc weight</td><td>BIC</td><td>SSE</td><td>MSE</td><td>RMSE</td><td>R-Square</td></tr><tr><td rowspan="2">Fp1</td><td>Logistic 4P</td><td>46127.267</td><td>1</td><td>46162.235</td><td>143802.21</td><td>17.841466</td><td>4.2239159</td><td>0.0371676</td></tr><tr><td>Gaussian Peak</td><td>46296.777</td><td>1.554e-37</td><td>46324.753</td><td>146893.49</td><td>18.222738</td><td>4.2688099</td><td>0.0164698</td></tr><tr><td rowspan="2">Fp2</td><td>Gaussian Peak</td><td>50977.192</td><td>0</td><td>51005.168</td><td>262464.63</td><td>32.55981</td><td>5.7061204</td><td>0.2190141</td></tr><tr><td>Logistic 4P</td><td>48867.636</td><td>1</td><td>48902.604</td><td>201999.73</td><td>25.062001</td><td>5.0061962</td><td>0.3989326</td></tr><tr><td rowspan="2">F3</td><td>Logistic 4P</td><td>48622.245</td><td>0.7085385</td><td>48657.214</td><td>195945.38</td><td>24.310841</td><td>4.9306025</td><td>0.0081122</td></tr><tr><td>Gaussian Peak</td><td>48624.022</td><td>0.2914615</td><td>48651.998</td><td>196037.23</td><td>24.31922</td><td>4.9314521</td><td>0.0076472</td></tr><tr><td rowspan="2">F4</td><td>Logistic 4P</td><td>59984.171</td><td>1</td><td>60019.14</td><td>801756.52</td><td>99.473513</td><td>9.9736409</td><td>0.550457</td></tr><tr><td>Gaussian Peak</td><td>63739.504</td><td>0</td><td>63767.48</td><td>1277607.4</td><td>158.49242</td><td>12.589377</td><td>0.2836485</td></tr><tr><td rowspan="2">F7</td><td>Logistic 4P</td><td>64325.669</td><td>1</td><td>64360.637</td><td>1373592.8</td><td>170.42094</td><td>13.054537</td><td>0.5533633</td></tr><tr><td>Gaussian Peak</td><td>68826.454</td><td>0</td><td>68854.429</td><td>2400822</td><td>297.83178</td><td>17.257804</td><td>0.21935</td></tr><tr><td rowspan="2">F8</td><td>Logistic 4P</td><td>47778.078</td><td>1</td><td>47813.047</td><td>176470.29</td><td>21.894577</td><td>4.6791641</td><td>0.8100189</td></tr><tr><td>Gaussian Peak</td><td>57959.343</td><td>0</td><td>57987.318</td><td>623880.25</td><td>77.394895</td><td>8.7974368</td><td>0.3283545</td></tr></table>

Table 14 Comparison of CNN with different time-frequency transforms.  

<table><tr><td>Time-Frequency representation with CNN</td><td>Accuracy</td></tr><tr><td>Wavelet Packet Decomposition (WPD) + CNN</td><td>85.17%</td></tr><tr><td>Short-Time Fourier Transform (STFT) + CNN</td><td>93.09%</td></tr><tr><td>Wavelet Transform (WT) + CNN</td><td>89.65%</td></tr><tr><td>Fast Fourier Transform (FFT) + CNN</td><td>90.65%</td></tr><tr><td>S-Transform, CNN with attention mechanism (Stockwell-CNN) Proposed</td><td>96.83%</td></tr></table>

by FTD. Furthermore, incorporating an attention mechanism enhances the model's performance by directing its focus toward the most relevant and informative EEG variables, thereby reducing the impact of irrelevant or noisy data. This selective emphasis on key features improves both the overall classification accuracy and the interpretability of the model. The combination of the Stockwell Transform, CNN, and the attention mechanism provides a comprehensive and robust approach to cognitive testing, enabling the early and reliable detection of Frontotemporal Dementia.

By improving both feature extraction and emphasis on important regions EEG data, the incorporation of attention mechanisms with the Stockwell Transform in FTD (Frontotemporal Dementia) classification greatly enhances model performance. By examining both spatial and frequency data, the Stockwell Transform offers time-frequency localization and a comprehensive depiction of EEG signal. When paired with attention processes, the model can improve classification accuracy by disregarding less significant features and prioritizing pertinent patterns and locations within the EEG data. This combination enhances the model's capability to distinguish FTD since other forms of dementia, such as AD, by enabling it to capture fine-grained information associated with the morphological and pathological aspects of FTD. In order to focus attention on the most informative regions of the EEG which are essential for diagnosing FTD the attention mechanism allows dynamic weighting of the spatial and spectral information.

CNN with various time frequency transforms for deep learning models for FTD is compared in Table 14. A comparison graph using Stockwell-CNN for cognitive testing is shown in Fig. 13. Advantages: (a) The Stockwell-CNN model with attention mechanisms enhances classification accuracy (96.83b) Focused Attention: The model uses the attention mechanism to emphasize and prioritize significant EEG signal locations, improving interpretability and identifying frontotemporal dementia (FTD) patterns. (c) Robust Performance: The technique is reliable for early dementia diagnosis since it is flexible and resilient across datasets and imaging circumstances.

When compared to other deep learning models, the Stockwell Transform performs better in FTD classification when attention processes are integrated. This method shows promise for an early and precise diagnosis of FTD by utilizing both time-frequency analysis and attention-based feature refinement, outperforming models that only use spatial feature extraction or conventional architectures. Nevertheless, problems with computational efficacy and possibly overfitting must be considered if one wants to maximize its application in therapeutic environments.

# 6. Discussion

Recent studies have explored transformer-based and hybrid deep learning (DL) architectures for improved detection of Alzheimer's Disease (AD) and cognitive impairments using neuroimaging and EEG data.

Yan et al. [15] introduced the Hybrid-RViT model, combining a Vision Transformer (ViT) with a ResNet-50 backbone for MRI-based AD classification. ResNet-50 aided in feature extraction via transfer learning, while ViT captured long-range dependencies through self-attention. This architecture achieved a training accuracy of  $97\%$  and a testing accuracy of  $95\%$ .

Shah et al. [16] proposed BiViT, a bidirectional ViT model incorporating a Parallel Coupled Encoding Strategy (PCES) and Mutual Latent Fusion (MLF). Evaluated across two diverse datasets, BiViT reached  $96.38\%$  accuracy in multi-class AD classification, including early and moderate cognitive impairments.

Shin et al. [17] applied ViT on 18F-Florbetaben PET images and compared it with CNN and VGG19. ViT outperformed VGG19 in binary classification (80% vs. 73.33%) but underperformed in three-class classification (56.67% vs. 66.67%). Interestingly, data augmentation degraded performance in both models, suggesting sensitivity to input variation.

Huang et al. [18] developed the Monte Carlo Ensemble Vision Transformer (MC-ViT), leveraging Monte Carlo sampling within a single ViT learner. Using the OASIS-3 and ADNI datasets, MC-ViT achieved  $90\%$  accuracy, outperforming both 2D and 3D CNN baselines while capturing 3D inter-feature correlations.

Beyond imaging, recent work demonstrates promising results using EEG-based deep learning. The Hybrid-RViT model attained  $96\%$  accuracy on EEG data, and BiViT preserved high performance  $(96.38\%)$  by modeling spatiotemporal features. Shin's CNN-ViT reached only  $80\%$ , indicating challenges with EEG spatial encoding. In contrast, MC-ViT remained robust with  $90\%$  accuracy on EEG signals.

Most notably, our proposed  $^{**}$ Stockwell-CNN $^{**}$  model, integrating the Stockwell Transform, CNN, and an attention mechanism, achieved  $^{**}96.83\%^{**}$  accuracy—surpassing existing transformer-based approaches. These findings highlight the advantage of combining time-frequency analysis with attention-driven deep networks for cognitive state classification as shown in Table 15

![](images/b817daf88fedf1b110628f1d3c97dcc8b214c5837a749b01e66012c4b46fa4bb.jpg)  
Fig. 13. Performance comparison of CNN models using various time-frequency transforms. Stockwell-CNN achieved the highest accuracy at  $96.83\%$ .

Table 15 Comparison of deep learning models for dementia diagnosis.  

<table><tr><td>S.No.</td><td>Model name</td><td>Accuracy</td><td>Author (Year)</td></tr><tr><td>1</td><td>Hybrid-RViT-ResNet-50</td><td>96%</td><td>Yan et al. [15]</td></tr><tr><td>2</td><td>BiViT</td><td>96.38%</td><td>Shah et al. [16]</td></tr><tr><td>3</td><td>CNN-ViT</td><td>80%</td><td>Shin et al. [17]</td></tr><tr><td>4</td><td>MC-ViT</td><td>90%</td><td>Huang et al. [18]</td></tr><tr><td>5</td><td>Stockwell-CNN (Proposed)</td><td>96.83%</td><td>Our study</td></tr></table>

# 7. Conclusion

The proposed Stockwell-CNN model, integrating the Stockwell Transform, CNNs, and an attention mechanism, demonstrates exceptional efficacy in EEG-based Frontotemporal Dementia (FTD) detection, achieving a classification accuracy of  $96.83\%$ . By synergizing time-frequency analysis with deep learning, the model captures dynamic brain activity patterns critical for distinguishing between normal cognition, moderate dementia, and severe FTD stages. The attention mechanism enhances interpretability by identifying diagnostically salient EEG features, bridging the gap between computational decision-making and clinical relevance. While the model outperforms traditional spectral methods (e.g., WT, STFT, FFT) and shows promise for real-time deployment, its computational demands and reliance on robust datasets highlight avenues for future optimization. This work underscores the transformative potential of hybrid deep learning frameworks in advancing early, accurate, and interpretable dementia diagnostics, paving the way for improved clinical decision-making and patient outcomes in neurodegenerative care.

# 8. Limitation and future enhancement

While the Stockwell-CNN model with attention mechanisms demonstrates strong performance in EEG-based FTD detection, it faces limitations, including computational complexity due to the integration of Stockwell Transform and attention layers, which increases memory and processing demands. Additionally, reliance on small or non-diverse datasets risks overfitting, potentially limiting generalizability to unseen populations or noisy EEG data. To address these challenges, future

work will prioritize collaborations with medical institutions to expand dataset diversity and scale, alongside advanced augmentation strategies like synthetic signal generation to enhance robustness. Computational efficiency can be improved through lightweight architectural optimizations, such as model pruning and quantization, without sacrificing accuracy. Incorporating explainable AI (XAI) methods will further clarify the attention mechanism's role in identifying diagnostic biomarkers, while integrating multimodal data (e.g., MRI, PET, or genetic markers) could refine diagnostic precision. Moreover, future studies could incorporate additional bio-signals (e.g., ECG, GSR) and emotional signals (e.g., facial expressions, speech patterns) to develop a multimodal diagnostic system, improving sensitivity in early mental health and FTD detection. Finally, developing user-friendly interfaces and conducting longitudinal clinical trials will bridge the gap between technical innovation and real-world clinical deployment, ensuring the model's practicality, ethical compliance, and scalability in improving early FTD diagnosis and patient care.

# CRediT authorship contribution statement

Siwei Xie: Investigation, Formal analysis, Data curation, Conceptualization, Visualization, Validation. Li Xiao: Formal analysis. Haitao Huang: Data curation. Dayang Chen: Software. Haiman Guo: Visualization. Amar Jain: Formal analysis, Data curation.

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Data availability

Data will be made available on request.

# References

[1] B. Jiao, R. Li, H. Zhou, K. Qing, H. Liu, H. Pan, L. Shen, Neural biomarker diagnosis and prediction to mild cognitive impairment and Alzheimer's disease using EEG technology, Alzheimers Res. Ther. 15 (1) (2023) 32, http://dx.doi.org/10.1186/s13195-023-01181-1.  
[2] A. Tisher, A. Salardini, A comprehensive update on treatment of dementia, in: Seminars in Neurology, Vol. 39, (02) Thieme Medical Publishers, 2019, pp. 167-178, http://dx.doi.org/10.1055/s-0039-1683408.  
[3] S. Kocahan, Z. Dogan, Mechanisms of Alzheimers disease pathogenesis and prevention: the brain, neural pathology, N-methyl-D-aspartate receptors, tau protein and other risk factors, Clin. Psychopharmacol. Neurosci. 15 (1) (2017) 1, http://dx.doi.org/10.9758/cpn.2017.15.1.1.  
[4] Y. Khan, B. Kaushik, C. Chowdhary, G. Srivastava, Ensemble model for diagnostic classification of Alzheimer's disease based on brain anatomical magnetic resonance imaging, Diagnostics 12 (12) (2022) 3193, http://dx.doi.org/10.3390/diagnostics12123193.  
[5] J. Trammell, P. MacRae, G. Davis, D. Bergstedt, A. Anderson, The relationship of cognitive performance and the theta-alpha power ratio is age-dependent: an EEG study of short term memory and reasoning during task and resting-state in healthy young and old adults, Front. Aging Neurosci. 9 (2017) 364, http://dx.doi.org/10.3389/fnagi.2017.00364.  
[6] S. Finnigan, I. Robertson, Resting EEG theta power correlates with cognitive performance in healthy older adults, Psychophysiology 48 (8) (2011) 1083-1087, http://dx.doi.org/10.1111/j.1469-8986.2010.01173.x.  
[7] D. Astle, J. Holmes, R. Kievitt, S. Gathercole, Annual Research Review: The transdiagnostic revolution in neurodevelopmental disorders, J. Child Psychol. Psychiatry 63 (4) (2022) 397-417, http://dx.doi.org/10.1111/jcppp.13481.  
[8] Y. Ma, J. Bland, T. Fujinami, Classification of Alzheimer's Disease and frontotemporal dementia using electroencephalography to quantify communication between electrode pairs, Diagnostics 14 (19) (2024) 2189, http://dx.doi.org/10.3390/diagnostics14192189.  
[9] M. Nour, U. Senturk, K. Polat, A novel hybrid model in the diagnosis and classification of alzheimer's disease using EEG signals: Deep ensemble learning (DEL) approach, Biomed. Signal Process. Control. 89 (2024) 105751, http://dx.doi.org/10.1016/j.bspc.2023.105751.

[10] M. Rostamikia, Y. Sarbaz, S. Makouei, EEG-based classification of Alzheimer's disease and frontotemporal dementia: a comprehensive analysis of discriminative features, Cogn. Neurodyn. 18 (6) (2024) 3447-3462, http://dx.doi.org/10.1007/s11571-024-10152-7.  
[11] R. Jiang, X. Zheng, J. Sun, L. Chen, G. Xu, R. Zhang, Classification for Alzheimer's disease and frontotemporal dementia via resting-state electroencephalography-based coherence and convolutional neural network, Cogn. Neurodyn. 19 (1) (2025) 46, http://dx.doi.org/10.1007/s11571-025-10232-2.  
[12] A. Miltiadous, K. Tzimourta, T. Afrantou, P. Ioannidis, N. Grigoriadis, D. Tsalikakis, A. Tzallas, A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG, Data 8 (6) (2023) 95, http://dx.doi.org/10.3390/data8060095.  
[13] Y. Si, R. He, L. Jiang, D. Yao, H. Zhang, P. Xu, F. Li, Differentiating between Alzheimer's disease and frontotemporal dementia based on the resting-state multilayer EEG network, IEEE Trans. Neural Syst. Rehabil. Eng. 31 (2023) 4521-4527, http://dx.doi.org/10.1109/TNSRE.2023.3329174.  
[14] Z. Wang, A. Liu, J. Yu, P. Wang, Y. Bi, S. Xue, W. Zhang, The effect of aperiodic components in distinguishing Alzheimer's disease from frontotemporal dementia, Geroscience 46 (1) (2024) 751-768, http://dx.doi.org/10.1007/s11357-023-01041-8.  
[15] H. Yan, V. Mubonanyikuzo, T. Komolafe, L. Zhou, T. Wu, N. Wang, Hybrid-RViT: Hybridizing ResNet-50 and vision transformer for enhanced Alzheimer's disease detection, PLoS One 20 (2) (2025) e0318998, http://dx.doi.org/10.1371/journal.pone.0318998.  
[16] S. Shah, M. Khan, A. Rizwan, S. Jan, N. Samee, M. Jamjoom, Computer-aided diagnosis of Alzheimer's disease and neurocognitive disorders with multimodal Bi-Vision Transformer (BiViT), Pattern Anal. Appl. 27 (3) (2024) 76, http://dx.doi.org/10.1007/s10044-024-01297-6.  
[17] H. Shin, S. Jeon, Y. Seol, S. Kim, D. Kang, Vision transformer approach for classification of Alzheimer's disease using 18F-Florbetaben brain images, Appl. Sci. 13 (6) (2023) 3453, http://dx.doi.org/10.3390/app13063453.  
[18] F. Huang, A. Qiu, A.D.N. Initiative, Ensemble vision transformer for dementia diagnosis, IEEE J. Biomed. Heal. Inform. (2024) http://dx.doi.org/10.1109/JBHI.2024.3412812.
