---
layout: default
---

# Introduction

X-rays are one of the most common imaging modalities used in modern medicine. An X-ray film contains a two-dimensional transposition of three-dimensional radiodensity between the X-ray source and the film. The two-dimensional nature of X-rays creates certain limitations on interpretation. Computed tomography (CT) imaging, on the other hand, computes three-dimensional radiodensity volumes from a series of X-rays performed around the patient. CT imaging is used in cases where a plain X-ray film is insufficient. CT imaging is more expensive and exposes patients to much greater levels of radiation than standard X-ray imaging (around two orders of magnitude greater). In the case of CT imaging, the benefits that the extra dimensionality provided must be weighed against the risk of increasing a patient’s radiation exposure and the additional cost of the procedure. We would not expect a 3D reconstruction of a 2D X-ray image to match the fidelity of its corresponding CT volume. It may, however, be possible that a 3D reconstruction could be “good enough” to increase diagnostic accuracy for particular diseases, allowing the patient to avoid unnecessary exposure to CT-sourced radiation and the additional expense of that modality.

It is also worth noting that there are many situations where X-ray imaging is the only radiological imaging modality readily available. This can often be the case in field hospitals and areas of the developing world. A 3D reconstruction of an X-ray where no CT imaging is available may prove helpful in certain diagnostic situations (eg, trauma surgery).

Finally, current machine learning (ML) vision tasks (recognition, segmentation, etc.) on X-ray images treat the X-ray films as a 2D picture—even though it contains 3D data. When radiologists interpret X-rays, they use their internal knowledge of 3D human anatomy to guide their interpretation. They are in essence combining image inputs and priori knowledge of human anatomy to achieve their diagnosis. Intuition would suggest that providing some kind of 3D understanding of a 2D X-ray film (via the embeddings from a 3D reconstruction process of that X-ray film) could improve accuracy on standard machine learning image tasks like recognition and segmentation. While testing this hypothesis is outside the scope of our project, it is a potential avenue for future research.





