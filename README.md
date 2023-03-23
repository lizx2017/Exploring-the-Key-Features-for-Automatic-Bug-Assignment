# Exploring-the-Key-Features-for-Automatic-Bug-Assignment-An-Empirical-Study


## Abstract

In the past decades, there was ever a great upsurge in designing automatic approaches for the bug assignment problem. Motivated by manual analysis, researchers have focused on improving natural language process techniques to deeply understand bug reports. At the same time, they also made attempt to introduce more useful information (e.g. components, version) from bug reports for further improvements. However, with various features explored, none of these studies has made comparison and evaluation of the effectiveness of different features. As a result, in spite of dozens of previous studies, researchers have to measure the importance of candidate features from scratch when they design new approaches. To remedy this limitation, in this paper, we conduct an empirical and comprehensive study on the effect of different features. Based on the prior researches, we collect all the features that have been considered in their work. To evaluate the effectiveness of each feature, we design a general classification model and introduce a bidirectional feature selection strategy to calculate the benefit of adding or deleting a feature. The result is beyond our expectations, exhibiting that nominal features are more effective than texts. For further understanding, we explain the advantages of nominal features by analyzing the evaluated projects in detail. We also explore the limitations as well as potential of textual features via using advanced NLP techniques. At the end of our empirical study, we evaluate the assignment accuracy with our selected features. Compared to groups of features conducted in prior researches, our selected features exhibit 11%-25% improvements on accuracy in assignment for all evaluated projects. Our study reflects that the effectiveness of nominal features is underestimated in bug assignments. Besides, it also calls for well-designed NLP techniques to distill useful information from texts.

## Dataset
We have uploaded the datasets we used. Please download the dataset and import it.

## How to use
1. Import data and update the database connection information in python files. For example, in RQ1-classify.py, users are required to modify SQL connection information including 'user', 'password', 'host' and 'database':
"connector = tools.sql_connect(user=<user>, password=<password>, host=<localhost>, database=<database>)"

2. RQ1-classifi