---
layout: page
title: Overview
---


# State-of-the-art

No one has tried to solve this problem in the context of chest X-rays and chest CTs. However, there exist several works that can perform 2D-to-3D reconstruction in different but similar domains. Henzler et al. (2018) has performed 2D-to-3D volume reconstruction of mammalian crania using a Convolutional Neural Network (CNN) with an encoder-decoder structure. However, they focused on optimizing surface structure and the generalizability of the algorithm (reconstruction any angle, wide variety of species). In constructing 3D volume from 2D images, Jackson et al. (2017) proposed a Volumetric Regression CNN (VRN) built upon paired 2D images and 3D facial models or scans. Their model is able to reconstruct 3D human faces from a single photograph taken at any angle. Karade and Ravi (2015) used a “bone template reconfiguration” algorithm involving Kohonen self-organizing maps to simulate 3D surface geometry of femur from biplane X-ray images.

Since we are interested in 2D-to-3D reconstruction, our training dataset, consisting of 3D CT scans, would also need an equivalent of the input X-ray images. We plan to generate synthetic X-rays for each of the 3D CT scans. Moturu and Chang (2019) proposed a method to create synthetic frontal chest X-rays using ray-tracing and Beer’s Law from chest CT scans. Moturu and Chang’s research focused on designing a neural network that can detect lung cancer, thus their methods also involved randomized nodule generation. Teixeira et al. (2018) also presented a framework that generates synthetic X-ray images from the surface geometry of the thorax and abdomen. The resulting X-ray images were only intended to serve as an approximation of the true internal anatomy. Other than that, we have not found any other papers that addressed the generation of synthetic X-ray images from CT scans specifically in the chest region. Henzler et al. (2018) also generated synthetic x-rays, but their algorithm simply flattened the CT scans at various angles and did not consider the point-source aspect of X-ray imaging. The only paper we found that included the point-source aspect of X-ray imaging was that of Moturu and Chang.


