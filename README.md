# Masked-Enforcerer
6261-15983-ITAI-1378-Midterm 
Project Name & Team Members: Masked Enforcerer. Jose Moreno

Tier Selection (Tier 1, 2, or 3 + justification): Tier 1 because I am doing this on my own and its easier to coordinate and perform better.

Problem Statement (2–3 sentences): After the events of the year 2020, it has become necessary to better enforce public health by any means, one of which is by enforcing the usage of masks to reduce the propataion of airborne diseases. However, fully enforcing these measures proves difficult by relying on humans.

Solution Overview (2–3 sentences): Created an automated tool which by using computer vision, it will scan multiple images featuring individuals with masks and detect who is doing it right and who is not using masks as they should, if not lacking one.

Technical Approach (technique, model, framework): Object detection + classification, using a YOLOv8 model and PyTorch (GoogleColab) since these are readily available at no cost.

Dataset Plan (source, size, labels, link if public)

Metrics (primary + secondary)

Week-by-Week Plan

Resources Needed (compute, cost, APIs): Google Colab

Risks & Mitigation Table

Risk	Probability	Mitigation
Low accuracy	Medium	Use data augmentation
Missing data	High	Switch to Roboflow dataset
