######
echo "\nExperiment 4"
echo "------------"
######

echo "Keep userId(2), context_datetime(4), weatherContext_humidity(18), weatherContext_visibility(19),audioContext_volumeNotif(31),
networkContext_mobileConnected(56), networkContext_wifiIpAddress(65), proximityContext_proximity(72),
proximityContext_proximityStandardDeviation(73), accelerationContext_accelerationStandardDeviation(81) , artist(99) "
echo "\nAttributes selected by WEKA on basis of Temperature"
echo "-------------------------------------------------------------------------------------------------"
mkdir experiment_4


java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.Remove -i Classification_dataset/MMG_data_subset.arff \
	  -o experiment_4/exp_4_1.arff -V -R 2,4,18,19,31,56,65,72,73,81,99
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.instance.RemoveWithValues -i experiment_4/exp_4_1.arff \
	  -o experiment_4/exp_4_2.arff -S 0.0 -C last -L 195 # remove <unknown> values (id 195)
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.instance.RemoveFrequentValues -i experiment_4/exp_4_2.arff \
	  -o experiment_4/exp_4_3.arff -C last -N 20
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.instance.RemoveDuplicates -i experiment_4/exp_4_3.arff \
	  -o experiment_4/exp_4_4.arff
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.NominalToString -i experiment_4/exp_4_4.arff \
	  -o experiment_4/exp_4_5.arff -C first-last
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.filters.unsupervised.attribute.StringToNominal -i experiment_4/exp_4_5.arff \
	  -o experiment_4/exp_4_6.arff -R first-last

echo "Majority Voting (ZeroR)"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.rules.ZeroR -t experiment_4/exp_4_6.arff > experiment_4/results_ZeroR.txt

echo "J48"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.J48 -t experiment_4/exp_4_6.arff -x 5 -C 0.25 -M 2 > experiment_4/results_J48.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.J48 -t experiment_4/exp_4_6.arff -x 10 -C 0.25 -M 2 >> experiment_4/results_J48.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.J48 -t experiment_4/exp_4_6.arff -x 10 -C 0.1 -M 2 >> experiment_4/results_J48.txt

echo "IBk (kNN)"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_4/exp_4_6.arff -x 10 -K 1 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" > experiment_4/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_4/exp_4_6.arff -x 10 -K 5 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_4/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_4/exp_4_6.arff -x 10 -K 10 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_4/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_4/exp_4_6.arff -x 10 -K 15 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_4/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_4/exp_4_6.arff -x 5 -K 1 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_4/results_IBk.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.lazy.IBk -t experiment_4/exp_4_6.arff -x 10 -K 1 -W 0 \
	  -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" >> experiment_4/results_IBk.txt

echo "LMT"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_4/exp_4_6.arff  -I -1 -M 15 -W 0.0 > experiment_4/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_4/exp_4_6.arff  -I -1 -M 7 -W 0.0 >> experiment_4/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_4/exp_4_6.arff  -I -1 -M 30 -W 0.0 >> experiment_4/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_4/exp_4_6.arff  -I 10 -M 15 -W 0.0 >> experiment_4/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_4/exp_4_6.arff  -I 5 -M 15 -W 0.0 >> experiment_4/results_LMT.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.LMT -t experiment_4/exp_4_6.arff   -I 20 -M 15 -W 0.0 >> experiment_4/results_LMT.txt

echo "RandomTree"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomTree -t experiment_4/exp_4_6.arff -K 0 -M 1.0 -V 0.0010 -S 1 > experiment_4/results_rTree.txt
 java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomTree -t experiment_4/exp_4_6.arff -K 10 -M 1.0 -V 0.0010 -S 1 >> experiment_4/results_rTree.txt

echo "RandomForest"
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_4/exp_4_6.arff -I 5 -K 0 -S 1 -num-slots 1 > experiment_4/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_4/exp_4_6.arff -I 25 -K 0 -S 1 -num-slots 1 >> experiment_4/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_4/exp_4_6.arff -I 50 -K 0 -S 1 -num-slots 1 >> experiment_4/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_4/exp_4_6.arff -I 75 -K 0 -S 1 -num-slots 1 >> experiment_4/results_rForest.txt
java -cp "/Applications/weka-3-7-12-apple-jvm.app/Contents/Resources/Java/weka.jar" \
	 weka.classifiers.trees.RandomForest -t experiment_4/exp_4_6.arff -I 100 -K 0 -S 1 -num-slots 1 >> experiment_4/results_rForest.txt