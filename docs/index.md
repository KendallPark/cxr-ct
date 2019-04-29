---
layout: default
title: Home
---

CS766 - Computer Vision - Spring 2019

[Course Web Page](http://pages.cs.wisc.edu/~mohitg/courses/CS766/)


Team Members: Kendall Park, Lanston Chu, Mehmet Demirel, Ying Ji Chuang


# Problem Statement

X-rays are one of the most common imaging modalities used in modern medicine. It is a quick and painless diagnostic tools that allow for a wide variety of examinations and procedures. During an X-ray, electromagnetic waves are sent through the patient and inflect the patient's internal structure on the exposed photographic film, creating a 2D image. 

Computed tomography (CT) imaging, on the other hand, uses rotating X-ray equipment to produce a series of cross-sectional images of the body at a defined slice interval. The resulting images can be stacked together to produce a 3D radiodensities volume.

Compared to X-rays, CT scans can provide detailed information of the human body, eliminating problem of overlapping structures. Despite the greater benefit of its extra dimensionality, CT imaging is more expensive and exposes patient to a much greater levels of radiation than X-ray imaging.

However, X-ray imaging are usually the most readily available imaging modality instead of CT due to its cost. When radiologists interpret X-rays, they use their internal knowledge of 3D human anatomy to guide their interpretation. They are in essence combining image inputs and priori knowledge of human anatomy to achieve their diagnosis. Intuition would suggest that providing some kind of 3D understanding of a 2D X-ray film (via the embeddings from a 3D reconstruction process of that X-ray film) could improve accuracy on standard machine learning image tasks like recognition and segmentation. While testing this hypothesis is outside the scope of our project, it is a potential avenue for future research.

If 3D reconstruction from a single 2D X-ray image is possible, it could be sufficient to increase diagnostic accuracy for particular diseases, which could also allow patient to avoid unnecessary exposure to CT-sourced radiation. 






