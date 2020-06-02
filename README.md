# STC_ASVspoof2015
# Keras implementation of the paper [STC Anti-spoofing Systems for the ASVspoof 2015 Challenge](https://drive.google.com/file/d/1v0uTbP1cuONwsoXFKXDmCUa7Iupbyd7Z/view)

* Requirements:
  - keras
  - librosa
  - matplotlib
  - numpy
  - tensorflow
  - sklearn
  - pandas
  
  ## Description: 
  
  ![MWPC feature extraction process](/images/MWPC.png)
  
* Detailed time-frequency analysis of the speech signals in our countermeasures. For this purpose we used front-end features based on applying the wavelet-packet transform, adapted to the mel scale.

* Instead of the classical energy of the frequency sub-bands,here we applied the Teager Keiser Energy (TKE) Operator. TKE is more informative than classical sample energy. Moreover, it is a noise-robust parameter for speech signal.

* For extracted features decorrelation, we consistently applied principal component analysis to derive 12 coefficients. We called these features Mel Wavelet Packet Coefficients (MWPC) for short. Here we observe MWPC with its first and second-order derivatives.

* To derive the MWPC coefficients we also used a Hamming window function with the window length equal to 256 and 50% overlap.
