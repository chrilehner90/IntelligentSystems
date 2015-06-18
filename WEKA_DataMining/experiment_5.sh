######
echo "\nExperiment 5"
echo "------------"
######

echo "Keep all except title and duration"
echo "-------------------------------------------------------------------------------------------------"
mkdir experiment_5


java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.Remove -i Classification_dataset/MMG_data_subset.arff \
	  -o experiment_5/exp_5_1.arff -R 100,101
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.instance.RemoveWithValues -i experiment_5/exp_5_1.arff \
	  -o experiment_5/exp_5_2.arff -S 0.0 -C last -L 195 # remove <unknown> values (id 195)
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.instance.RemoveFrequentValues -i experiment_5/exp_5_2.arff \
	  -o experiment_5/exp_5_3.arff -C last -N 20
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.instance.RemoveDuplicates -i experiment_5/exp_5_3.arff \
	  -o experiment_5/exp_5_4.arff
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.NominalToString -i experiment_5/exp_5_4.arff \
	  -o experiment_5/exp_5_5.arff -C first-last
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.StringToNominal -i experiment_5/exp_5_5.arff \
	  -o experiment_5/exp_5_6.arff -R first-last

echo "Majority Voting (ZeroR)"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.rules.ZeroR -t experiment_5/exp_5_6.arff > experiment_5/results_ZeroR.txt

echo "J48"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.J48 -t experiment_5/exp_5_6.arff -x 5 -C 0.25 -M 2 > experiment_5/results_J48.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.J48 -t experiment_5/exp_5_6.arff -x 10 -C 0.25 -M 2 >> experiment_5/results_J48.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.J48 -t experiment_5/exp_5_6.arff -x 10 -C 0.1 -M 2 >> experiment_5/results_J48.txt

echo "IBk (kNN)"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_5/exp_5_6.arff -x 10 -K 1 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" > experiment_5/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_5/exp_5_6.arff -x 10 -K 5 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_5/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_5/exp_5_6.arff -x 10 -K 10 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_5/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_5/exp_5_6.arff -x 10 -K 15 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_5/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_5/exp_5_6.arff -x 5 -K 1 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_5/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_5/exp_5_6.arff -x 10 -K 1 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_5/results_IBk.txt

echo "LMT"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_5/exp_5_6.arff  -I -1 -M 15 -W 0.0 > experiment_5/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_5/exp_5_6.arff  -I -1 -M 7 -W 0.0 >> experiment_5/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_5/exp_5_6.arff  -I -1 -M 30 -W 0.0 >> experiment_5/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_5/exp_5_6.arff  -I 10 -M 15 -W 0.0 >> experiment_5/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_5/exp_5_6.arff  -I 5 -M 15 -W 0.0 >> experiment_5/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_5/exp_5_6.arff   -I 20 -M 15 -W 0.0 >> experiment_5/results_LMT.txt

echo "RandomTree"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomTree -t experiment_5/exp_5_6.arff -K 0 -M 1.0 -V 0.0010 -S 1 > experiment_5/results_rTree.txt
 java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomTree -t experiment_5/exp_5_6.arff -K 10 -M 1.0 -V 0.0010 -S 1 >> experiment_5/results_rTree.txt

echo "RandomForest"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_5/exp_5_6.arff -I 5 -K 0 -S 1 -num-slots 1 > experiment_5/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_5/exp_5_6.arff -I 25 -K 0 -S 1 -num-slots 1 >> experiment_5/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_5/exp_5_6.arff -I 50 -K 0 -S 1 -num-slots 1 >> experiment_5/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_5/exp_5_6.arff -I 75 -K 0 -S 1 -num-slots 1 >> experiment_5/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_5/exp_5_6.arff -I 100 -K 0 -S 1 -num-slots 1 >> experiment_5/results_rForest.txt