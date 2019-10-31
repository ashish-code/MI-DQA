# MI-DQA
Medical Imaging - Deep Quality Assessment

## Abstract
Structural Magnetic Resonance Imaging (MRI) is the standard-of-care imaging modality for screening and diagnosis of most neurological conditions. However, the ability to reliably assess brain MRIs for diagnostic purposes is often hampered due to the presence of artifacts associated with magnetic field inhomogeneity, aliasing, as well as other patient-related motion artifacts. Further, some machine-related artifacts are not visually obvious but severely affect the quality of the scans. Reliable quality control (QC) of brain MRI scans would allow for precluding erroneous acquisitions and reducing biases in subsequent diagnosis.  Since visual inspection is impractical for large volumes of data and subject to inter-observer variability, automated QC could serve as an excellent automated tool for radiological assessments of brain MRI scans. Existing state-of-art automated methods for QC utilize a combination of (a) hand-crafted image features, and (b) standard machine learning algorithms including Support Vector Machines (SVM) and Random Forests (RF); both of which have been rendered obsolete by Deep Learning in various fields related to image analysis. In this invention a novel deep learning based methodology is created for QC in MRI. It provides a system and method for QC that is: (a) state-of-art in performance; (b) extensible to new MRI scans using transfer learning; and (c) scalable to ever increasing volumes of MRI data.

### Prior Art: 
1. Oscar Esteban, Daniel Birman, Marie Schaer, Oluwasanmi O Koyejo, Russell A Poldrack, and Krzysztof J Gorgolewski. MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites. PloS one, 12(9):1-21, 2017.
2. Jerey P. Woodard and Monica P. Carley-Spencer. No-Reference Image Quality Metrics for Structural MRI. Neuroinformatics, 4(3):243-262, 2006.

## Novel Features:
    (1) This invention uses fully connected Dense Neural Networks (DNN) instead of SVMs, and RFs, to classify the quality assessment label for hand-crafted Image Quality Metrics (IQMs) features suggested by the Perceptual Connectomes Project (PCP)’s Quality Assessment Protocol (QAP). DNNs are extensible and transferable to both new MRI data, types of artifacts in MRI affecting quality and new IQM features. Prior-Art methods all require computationally expensive re-training of their model from scratch, which renders them obsolete in the era of big data.
    (2) This invention incorporates Residual Network (ResNet-18) deep learning based model for visual feature extraction directly from MRI scans. The superiority of deep learning as a feature extractor in comparison to hand-crafted features has been empirically established in every application domain of image analysis. The proposed model is extensible and scalable to new MRI data and new types of artifacts.
    
## Proof of Principle:
    (1) Data: Publicly available ABIDE-1 and DS030 MRI data was solely utilized in the development of this invention. ABIDE-1 is a multi-site dataset from which we used 15 sites for training our model and 2 sites as hold-out for validation. The DS030 is from 2 sites, which was utilized exclusively for testing our trained model.
    (2) DNN: A fully connected model with multiple blocks of linear transformation, batch normalization, ReLU activation and Dropout regularization was developed after extensive empirically driven configuration optimization towards a unique network architecture specific to QC for MRI using IQMs. The model architecture; experimental workflow block diagram; model training performance measures are illustrated below.
 ![IQA using DNN](https://www.dropbox.com/s/lo4iv76ff00dw1v/dnn.png?raw=1)
    (3) Residual Network: A ResNet-18 based model is incorporated into our novel MRI artifact feature extraction method for QC. The proposed model is shown to automatically learn relevant low-level visual features that lead to superior performance for QC. The hyper-parameters of the network and methodology of data processing is determined after extensive experimentation and in conjunction with the architecture constitute a novel invention. The model architecture, performance analysis and visualization of the learning artifacts in MRI are illustrated below.
![IQA using ResNet](https://www.dropbox.com/s/pvzcfl8jaogua7y/resnet.png?raw=1)

We prove that our network is learning to distinguish between pristine MRI and MRI slices with artifacts, we show the gradient class activmation map based heatmap for random patches selected from MRI data that is rated by experts to be clean (top row) and have artifacts (bottom row). It seems evident that the network has learned to place emphasis on regions of MRI that have some artifacts in them. A sample is shown here:
![Grad-CAM of clean and artifact MRI slices](https://www.dropbox.com/s/3v42cdna33c6xnh/networklearning.png?raw=1)
