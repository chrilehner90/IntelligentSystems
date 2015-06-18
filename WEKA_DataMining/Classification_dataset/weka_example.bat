REM apply filters
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.filters.unsupervised.attribute.Remove -i MMG_data_subset.arff -o MMG_3attribs.arff -V -R 6,7,99
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.filters.unsupervised.instance.RemoveFrequentValues -i MMG_3attribs.arff -o MMG_3attribs_2.arff -C last -N 5 
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.filters.unsupervised.attribute.NominalToString -i MMG_3attribs_2.arff -o MMG_3attribs_3.arff -C last
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.filters.unsupervised.attribute.StringToNominal -i MMG_3attribs_3.arff -o MMG_3attribs_4.arff -R last

REM J48
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.trees.J48 -t MMG_3attribs_4.arff -x 5 -i -C 0.25 -M 2 > results_J48.txt
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.trees.J48 -t MMG_3attribs_4.arff -x 10 -i -C 0.25 -M 2 >> results_J48.txt
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.trees.J48 -t MMG_3attribs_4.arff -x 10 -i -C 0.1 -M 2 >> results_J48.txt

REM IBk
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.lazy.IBk -t MMG_3attribs_4.arff -x 10 -i -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" > results_IBk.txt
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.lazy.IBk -t MMG_3attribs_4.arff -x 10 -i -K 5 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> results_IBk.txt
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.lazy.IBk -t MMG_3attribs_4.arff -x 10 -i -K 15 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> results_IBk.txt

REM SMO
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.functions.SMO -t MMG_3attribs_4.arff -x 3 -i -C 1.0 -L 0.001 -P 1.0E-12 -N 2 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.Puk -C 250007 -O 1.0 -S 1.0"  > results_SMO.txt
java -cp "C:/Program Files/Weka-3-6/weka.jar" weka.classifiers.functions.SMO -t MMG_3attribs_4.arff -x 10 -i -C 1.0 -L 0.001 -P 1.0E-12 -N 2 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.Puk -C 250007 -O 1.0 -S 1.0"  >> results_SMO.txt
