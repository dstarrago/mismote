# mismote
Class imbalance correction algorithm for multiple-instance data

In contrasts with regular classification problems, in which each example has a unique description, in multiple-instance classification (MIC) problems, each example has many descriptions. In the same way as regular classification problems, multiple-instances classification problems can suffer from the class imbalance problem. A data set suffer from the class imbalance problem when one or more of its classes are underrepresented, which means that the size of these classes is much smaller than that of the rest of the classes. Underrepresented classes are hard to learn by classification algorithms, and their instances are frequently misclassified in favor of the larger classes.

A successful method to deal with the class imbalance problem in regular data classification is called <a href="https://jair.org/index.php/jair/article/view/10302" target="_blank">SMOTE</a>. SMOTE generates synthetic examples in an underrepresented class through interpolation of examples that are present in that class. With MISMOTE, we brought SMOTE's idea to multiple-instance classification, creating synthetic bags in underrepresented classes. 

You can find a complete description of MISMOTE and its experimental results in 
- Tarragó, D.S., Cornelis, C., Bello, R., Herrera, F.: MISMOTE: synthetic minority over-sampling technique for multiple instance learning with imbalanced data. Central University Marta Abreu de Las Villas (2014). <a href="https://www.researchgate.net/publication/332513537_MISMOTE_synthetic_minority_over-sampling_technique_for_multiple_instance_learning_with_imbalanced_data" target="_blank">(text)</a>

Developed with:
- Java 1.8
- NetBeans IDE 8.2

Dependencies:
- Weka 3.7
